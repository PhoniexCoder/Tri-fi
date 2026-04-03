"""
csi_frame_parser.py — ADR-018 Binary Frame Decoder
Parses raw UDP frames sent by csi_collector.c into structured numpy arrays.

ADR-018 Frame Layout (from csi_collector.h):
  Bytes [0..3]   Magic: 0xC5110001 (LE u32)
  Byte  [4]      Node ID
  Byte  [5]      N antennas
  Bytes [6..7]   N subcarriers (LE u16)  = iq_len / (2 * n_ant)
  Bytes [8..11]  Frequency MHz (LE u32)
  Bytes [12..15] Sequence number (LE u32)
  Byte  [16]     RSSI (i8)
  Byte  [17]     Noise floor (i8)
  Bytes [18..19] Reserved
  Bytes [20..]   I/Q data: alternating int8 [real0, imag0, real1, imag1, ...]

ADR-018 Vitals Packet (magic 0xC5110002) is produced by edge_processing.c
and carries pre-computed vitals — this parser handles BOTH frame types.
"""

import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── Constants from csi_collector.h ────────────────────────────────────────────
CSI_MAGIC_RAW    = 0xC5110001   # Raw I/Q frame
CSI_MAGIC_VITALS = 0xC5110002   # Pre-computed vitals packet
CSI_HEADER_SIZE  = 20


@dataclass
class CSIFrame:
    """
    Decoded ADR-018 raw I/Q frame.

    amplitude  : float32 array (n_subcarriers,) — |I + jQ| per subcarrier
    phase      : float32 array (n_subcarriers,) — atan2(Q, I) per subcarrier
    phase_diff : float32 array (n_subcarriers-1,) — adjacent phase differences
                 (removes carrier-frequency offset, more robust for sensing)
    rssi       : int dBm
    n_subcarriers: int  (typically 64 or 128 for HT20/HT40)
    node_id    : int
    seq        : int  sequence counter
    freq_mhz   : int  WiFi channel centre frequency
    """
    amplitude   : np.ndarray
    phase       : np.ndarray
    phase_diff  : np.ndarray
    rssi        : int
    noise_floor : int
    n_subcarriers: int
    node_id     : int
    seq         : int
    freq_mhz    : int
    raw_iq      : np.ndarray   # raw int8 I/Q kept for debugging


@dataclass
class VitalsPacket:
    """Decoded ADR-018 vitals packet (magic 0xC5110002)."""
    breathing_bpm : float
    heart_bpm     : float
    motion_energy : float
    presence      : bool
    node_id       : int
    seq           : int


class CSIFrameParser:
    """
    Parses raw UDP bytes from the ESP32-S3 into CSIFrame or VitalsPacket objects.

    Usage:
        parser = CSIFrameParser()
        frame = parser.parse(udp_payload_bytes)
        if isinstance(frame, CSIFrame):
            amp = frame.amplitude      # shape: (n_sub,)
            ph  = frame.phase          # shape: (n_sub,)
    """

    def __init__(self, expected_node_id: Optional[int] = None):
        """
        expected_node_id: if set, frames from other nodes are silently dropped.
        """
        self._expected_node = expected_node_id
        self.n_parsed  = 0
        self.n_dropped = 0

    def parse(self, data: bytes):
        """
        Parse a UDP payload.
        Returns CSIFrame | VitalsPacket | None (on parse error / unknown magic).
        """
        if len(data) < CSI_HEADER_SIZE:
            self.n_dropped += 1
            return None

        magic = struct.unpack_from("<I", data, 0)[0]

        if magic == CSI_MAGIC_RAW:
            return self._parse_raw(data)
        elif magic == CSI_MAGIC_VITALS:
            return self._parse_vitals(data)
        else:
            self.n_dropped += 1
            return None

    # ── private ───────────────────────────────────────────────────────────────

    def _parse_raw(self, data: bytes) -> Optional[CSIFrame]:
        node_id      = data[4]
        n_ant        = data[5]
        n_sub        = struct.unpack_from("<H", data, 6)[0]
        freq_mhz     = struct.unpack_from("<I", data, 8)[0]
        seq          = struct.unpack_from("<I", data, 12)[0]
        rssi         = struct.unpack_from("b",  data, 16)[0]   # signed
        noise_floor  = struct.unpack_from("b",  data, 17)[0]

        if self._expected_node is not None and node_id != self._expected_node:
            self.n_dropped += 1
            return None

        iq_bytes = data[CSI_HEADER_SIZE:]
        if len(iq_bytes) < 2:
            self.n_dropped += 1
            return None

        # Raw I/Q: alternating int8 [I0 Q0 I1 Q1 ...]
        raw = np.frombuffer(iq_bytes, dtype=np.int8).astype(np.float32)

        # Handle odd-length (should never happen, but guard anyway)
        if len(raw) % 2 != 0:
            raw = raw[:-1]

        I = raw[0::2]
        Q = raw[1::2]

        amplitude  = np.sqrt(I**2 + Q**2)               # (n_sub,)
        phase      = np.arctan2(Q, I)                    # (n_sub,) in [-π, π]
        phase_diff = np.diff(np.unwrap(phase))           # (n_sub-1,) CFO-free

        self.n_parsed += 1
        return CSIFrame(
            amplitude    = amplitude,
            phase        = phase,
            phase_diff   = phase_diff,
            rssi         = rssi,
            noise_floor  = noise_floor,
            n_subcarriers= len(amplitude),
            node_id      = node_id,
            seq          = seq,
            freq_mhz     = freq_mhz,
            raw_iq       = raw,
        )

    def _parse_vitals(self, data: bytes) -> Optional[VitalsPacket]:
        """
        Parse 0xC5110002 vitals packet from edge_processing.c.
        Layout (32 bytes total, see edge_processing.c broadcast_vitals_packet):
          [0..3]  magic
          [4]     node_id
          [5]     reserved
          [6..7]  seq (LE u16)
          [8..11] breathing_bpm (LE f32)
          [12..15] heart_bpm (LE f32)
          [16..19] motion_energy (LE f32)
          [20]    presence (u8, 0/1)
        """
        if len(data) < 21:
            return None
        node_id      = data[4]
        seq          = struct.unpack_from("<H", data, 6)[0]
        br           = struct.unpack_from("<f", data, 8)[0]
        hr           = struct.unpack_from("<f", data, 12)[0]
        motion       = struct.unpack_from("<f", data, 16)[0]
        presence     = bool(data[20])
        return VitalsPacket(
            breathing_bpm = float(br),
            heart_bpm     = float(hr),
            motion_energy = float(motion),
            presence      = presence,
            node_id       = node_id,
            seq           = int(seq),
        )
