"""
csi_frame_parser.py — decoder for the UDP packets sent by the ESP32-S3 CSI node.

Every packet starts with a little-endian u32 magic. The layouts below mirror
the C definitions exactly; tests/test_csi_frame_parser.py round-trips them
against the real firmware serializer and structs.

  0xC5110001  Raw CSI frame     csi_collector.c  csi_serialize_frame()
  0xC5110002  Vitals (32 B)     edge_processing.h edge_vitals_pkt_t
  0xC5110003  Compressed CSI    edge_processing.c (delta + RLE; skipped here)
  0xC5110004  Fused vitals (48 B, CSI + mmWave) edge_fused_vitals_pkt_t
  0xC5110005  WASM module output wasm_runtime.h (skipped here)
  0xC5110006  Anchor beacon     anchor.c (ESP-NOW payload; never sent over UDP)

Raw CSI frame header (20 bytes):
  [0..3]   magic u32
  [4]      node_id u8
  [5]      n_antennas u8
  [6..7]   n_subcarriers u16          (per antenna)
  [8..11]  freq_mhz u32
  [12..15] sequence u32
  [16]     rssi i8
  [17]     noise_floor i8
  [18..19] reserved
  [20..]   CSI bytes exactly as delivered by ESP-IDF: for each subcarrier,
           one signed byte imaginary part followed by one signed byte real part.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

# ── Magic numbers (keep in sync with the firmware headers) ──────────────────
CSI_MAGIC_RAW        = 0xC5110001
CSI_MAGIC_VITALS     = 0xC5110002
CSI_MAGIC_COMPRESSED = 0xC5110003
CSI_MAGIC_FUSED      = 0xC5110004
CSI_MAGIC_WASM       = 0xC5110005
CSI_MAGIC_ANCHOR     = 0xC5110006

CSI_HEADER_SIZE = 20

# struct formats mirroring the packed C structs
RAW_HEADER_FMT = "<IBBHIIbbH"            # 20 bytes
VITALS_FMT     = "<IBBHIbB2xffII"        # 32 bytes, edge_vitals_pkt_t
FUSED_FMT      = "<IBBHIbBBBffIfffBBHI"  # 48 bytes, edge_fused_vitals_pkt_t

VITALS_SIZE = struct.calcsize(VITALS_FMT)
FUSED_SIZE  = struct.calcsize(FUSED_FMT)
assert struct.calcsize(RAW_HEADER_FMT) == CSI_HEADER_SIZE
assert VITALS_SIZE == 32 and FUSED_SIZE == 48

# Vitals flag bits (edge_vitals_pkt_t.flags)
FLAG_PRESENCE = 0x01
FLAG_FALL     = 0x02
FLAG_MOTION   = 0x04
FLAG_MMWAVE   = 0x08   # fused packets only

_SKIPPED_MAGICS = {CSI_MAGIC_COMPRESSED, CSI_MAGIC_WASM, CSI_MAGIC_ANCHOR}


@dataclass
class CSIFrame:
    """Decoded raw CSI frame (antenna 0)."""
    amplitude    : np.ndarray   # float32 (n_subcarriers,)  |real + j*imag|
    phase        : np.ndarray   # float32 (n_subcarriers,)  atan2(imag, real)
    phase_diff   : np.ndarray   # float32 (n_subcarriers-1,) adjacent differences
    rssi         : int          # dBm
    noise_floor  : int          # dBm
    n_subcarriers: int
    n_antennas   : int
    node_id      : int
    seq          : int
    freq_mhz     : int
    raw_iq       : np.ndarray   # int8 bytes as received (imag, real pairs)


@dataclass
class VitalsPacket:
    """Decoded on-chip vitals packet (0xC5110002)."""
    node_id       : int
    flags         : int
    breathing_bpm : float
    heart_bpm     : float
    rssi          : int
    n_persons     : int
    motion_energy : float
    presence_score: float
    timestamp_ms  : int

    @property
    def presence(self) -> bool:
        return bool(self.flags & FLAG_PRESENCE)

    @property
    def fall(self) -> bool:
        return bool(self.flags & FLAG_FALL)

    @property
    def motion(self) -> bool:
        return bool(self.flags & FLAG_MOTION)


@dataclass
class FusedVitalsPacket(VitalsPacket):
    """Decoded fused CSI + mmWave vitals packet (0xC5110004)."""
    mmwave_type      : int = 0
    fusion_confidence: int = 0
    mmwave_hr_bpm    : float = 0.0
    mmwave_br_bpm    : float = 0.0
    mmwave_distance  : float = 0.0   # cm
    mmwave_targets   : int = 0
    mmwave_confidence: int = 0

    @property
    def mmwave_present(self) -> bool:
        return bool(self.flags & FLAG_MMWAVE)


@dataclass
class LinkStats:
    """Per-node packet accounting for raw CSI frames."""
    frames   : int = 0
    lost     : int = 0
    last_seq : Optional[int] = None
    last_rssi: int = 0

    @property
    def loss_ratio(self) -> float:
        total = self.frames + self.lost
        return self.lost / total if total else 0.0


ParseResult = Optional[Union[CSIFrame, VitalsPacket]]


class CSIFrameParser:
    """
    Parses UDP payloads from ESP32-S3 nodes.

        parser = CSIFrameParser()
        result = parser.parse(payload)
        if isinstance(result, CSIFrame): ...
        elif isinstance(result, VitalsPacket): ...   # includes FusedVitalsPacket
    """

    def __init__(self, expected_node_id: Optional[int] = None):
        self._expected_node = expected_node_id
        self.n_parsed  = 0
        self.n_dropped = 0   # malformed or unknown
        self.n_skipped = 0   # valid but not decoded here (compressed, WASM)
        self.stats: dict[int, LinkStats] = {}

    def parse(self, data: bytes) -> ParseResult:
        if len(data) < 4:
            self.n_dropped += 1
            return None

        magic = struct.unpack_from("<I", data, 0)[0]
        if magic == CSI_MAGIC_RAW:
            result = self._parse_raw(data)
        elif magic == CSI_MAGIC_VITALS:
            result = self._parse_vitals(data)
        elif magic == CSI_MAGIC_FUSED:
            result = self._parse_fused(data)
        elif magic in _SKIPPED_MAGICS:
            self.n_skipped += 1
            return None
        else:
            result = None

        if result is None:
            self.n_dropped += 1
            return None
        if self._expected_node is not None and result.node_id != self._expected_node:
            self.n_dropped += 1
            return None
        self.n_parsed += 1
        return result

    # ── private ───────────────────────────────────────────────────────────────

    def _parse_raw(self, data: bytes) -> Optional[CSIFrame]:
        if len(data) < CSI_HEADER_SIZE:
            return None
        (_, node_id, n_ant, n_sub, freq_mhz, seq,
         rssi, noise_floor, _) = struct.unpack_from(RAW_HEADER_FMT, data, 0)
        if n_ant == 0 or n_sub == 0:
            return None

        expected = n_ant * n_sub * 2
        iq_bytes = data[CSI_HEADER_SIZE:]
        if len(iq_bytes) < expected:
            return None   # truncated frame
        raw = np.frombuffer(iq_bytes[:expected], dtype=np.int8)

        # Antenna 0 only; ESP-IDF stores (imaginary, real) per subcarrier.
        ant0 = raw[: n_sub * 2].astype(np.float32)
        imag = ant0[0::2]
        real = ant0[1::2]

        amplitude  = np.sqrt(real ** 2 + imag ** 2)
        phase      = np.arctan2(imag, real).astype(np.float32)
        phase_diff = np.diff(np.unwrap(phase)).astype(np.float32)

        self._account(node_id, seq, rssi)
        return CSIFrame(
            amplitude=amplitude, phase=phase, phase_diff=phase_diff,
            rssi=rssi, noise_floor=noise_floor,
            n_subcarriers=n_sub, n_antennas=n_ant,
            node_id=node_id, seq=seq, freq_mhz=freq_mhz, raw_iq=raw,
        )

    def _account(self, node_id: int, seq: int, rssi: int) -> None:
        st = self.stats.setdefault(node_id, LinkStats())
        if st.last_seq is not None:
            gap = (seq - st.last_seq) & 0xFFFFFFFF
            if 1 < gap < 10_000:          # ignore reorders and node reboots
                st.lost += gap - 1
        st.last_seq = seq
        st.frames += 1
        st.last_rssi = rssi

    @staticmethod
    def _parse_vitals(data: bytes) -> Optional[VitalsPacket]:
        if len(data) < VITALS_SIZE:
            return None
        (_, node_id, flags, br_x100, hr_x10000, rssi, n_persons,
         motion, presence_score, ts_ms, _) = struct.unpack_from(VITALS_FMT, data, 0)
        return VitalsPacket(
            node_id=node_id, flags=flags,
            breathing_bpm=br_x100 / 100.0, heart_bpm=hr_x10000 / 10000.0,
            rssi=rssi, n_persons=n_persons,
            motion_energy=float(motion), presence_score=float(presence_score),
            timestamp_ms=ts_ms,
        )

    @staticmethod
    def _parse_fused(data: bytes) -> Optional[FusedVitalsPacket]:
        if len(data) < FUSED_SIZE:
            return None
        (_, node_id, flags, br_x100, hr_x10000, rssi, n_persons,
         mw_type, fusion_conf, motion, presence_score, ts_ms,
         mw_hr, mw_br, mw_dist, mw_targets, mw_conf, _, _) = struct.unpack_from(FUSED_FMT, data, 0)
        return FusedVitalsPacket(
            node_id=node_id, flags=flags,
            breathing_bpm=br_x100 / 100.0, heart_bpm=hr_x10000 / 10000.0,
            rssi=rssi, n_persons=n_persons,
            motion_energy=float(motion), presence_score=float(presence_score),
            timestamp_ms=ts_ms,
            mmwave_type=mw_type, fusion_confidence=fusion_conf,
            mmwave_hr_bpm=float(mw_hr), mmwave_br_bpm=float(mw_br),
            mmwave_distance=float(mw_dist), mmwave_targets=mw_targets,
            mmwave_confidence=mw_conf,
        )
