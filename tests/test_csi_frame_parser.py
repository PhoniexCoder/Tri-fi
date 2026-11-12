"""Decoder tests: Python must read exactly what the firmware writes."""

import re
import shutil
import struct
import subprocess

import numpy as np
import pytest

from conftest import FIRMWARE_DIR
from csi_frame_parser import (
    CSI_HEADER_SIZE, CSI_MAGIC_ANCHOR, CSI_MAGIC_COMPRESSED, CSI_MAGIC_FUSED,
    CSI_MAGIC_RAW, CSI_MAGIC_VITALS, CSI_MAGIC_WASM, FUSED_FMT, VITALS_FMT,
    CSIFrame, CSIFrameParser, FusedVitalsPacket, VitalsPacket,
)

TEST_DIR = FIRMWARE_DIR / "test"
MAIN_DIR = FIRMWARE_DIR / "main"


# ── Helpers ──────────────────────────────────────────────────────────────────

def raw_frame(node_id=1, seq=0, n_sub=64, n_ant=1, rssi=-50, iq=None):
    if iq is None:
        iq = bytes(range(n_sub * n_ant * 2))
    header = struct.pack("<IBBHIIbbH", CSI_MAGIC_RAW, node_id, n_ant, n_sub,
                         2437, seq, rssi, -90, 0)
    return header + iq


def vitals_packet(**kw):
    v = dict(node_id=2, flags=0x01, br=1450, hr=722500, rssi=-61, n=1,
             motion=0.375, presence=0.8125, ts=123456)
    v.update(kw)
    return struct.pack(VITALS_FMT, CSI_MAGIC_VITALS, v["node_id"], v["flags"], v["br"],
                       v["hr"], v["rssi"], v["n"], v["motion"], v["presence"], v["ts"], 0)


# ── Round trip against the real firmware code ────────────────────────────────

@pytest.fixture(scope="module")
def firmware_packets(tmp_path_factory):
    cc = shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no host C compiler available")
    exe = tmp_path_factory.mktemp("wire") / "wire_dump"
    subprocess.run(
        [cc, "-std=gnu11", "-I", str(TEST_DIR / "stubs"), "-I", str(MAIN_DIR),
         str(TEST_DIR / "wire_dump.c"), str(MAIN_DIR / "csi_collector.c"),
         str(TEST_DIR / "stubs" / "esp_stubs.c"), "-o", str(exe), "-lm"],
        check=True, capture_output=True, text=True,
    )
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    return {tag: bytes.fromhex(hx) for tag, hx in (line.split() for line in out.splitlines())}


def test_raw_frame_matches_firmware_serializer(firmware_packets):
    parser = CSIFrameParser()
    frame = parser.parse(firmware_packets["raw"])
    assert isinstance(frame, CSIFrame)
    assert (frame.node_id, frame.n_antennas, frame.n_subcarriers) == (3, 1, 64)
    assert (frame.rssi, frame.noise_floor, frame.freq_mhz, frame.seq) == (-47, -92, 2437, 0)

    iq = np.array([(i * 7 - 60) for i in range(128)], dtype=np.int64).astype(np.int8).astype(np.float32)
    imag, real = iq[0::2], iq[1::2]
    np.testing.assert_allclose(frame.amplitude, np.hypot(real, imag), rtol=1e-6)
    np.testing.assert_allclose(frame.phase, np.arctan2(imag, real), rtol=1e-6)

    second = parser.parse(firmware_packets["raw2"])
    assert second.seq == 1
    assert parser.stats[3].frames == 2 and parser.stats[3].lost == 0


def test_vitals_matches_firmware_struct(firmware_packets):
    pkt = CSIFrameParser().parse(firmware_packets["vitals"])
    assert type(pkt) is VitalsPacket
    assert pkt.node_id == 2
    assert pkt.breathing_bpm == pytest.approx(14.5)
    assert pkt.heart_bpm == pytest.approx(72.25)
    assert (pkt.rssi, pkt.n_persons, pkt.timestamp_ms) == (-61, 1, 123456)
    assert pkt.motion_energy == pytest.approx(0.375)
    assert pkt.presence_score == pytest.approx(0.8125)
    assert pkt.presence and pkt.motion and not pkt.fall


def test_fused_vitals_matches_firmware_struct(firmware_packets):
    pkt = CSIFrameParser().parse(firmware_packets["fused"])
    assert isinstance(pkt, FusedVitalsPacket) and isinstance(pkt, VitalsPacket)
    assert pkt.node_id == 1
    assert pkt.breathing_bpm == pytest.approx(15.25)
    assert pkt.heart_bpm == pytest.approx(71.5)
    assert (pkt.rssi, pkt.n_persons, pkt.timestamp_ms) == (-55, 2, 654321)
    assert (pkt.mmwave_type, pkt.fusion_confidence) == (2, 90)
    assert pkt.mmwave_distance == pytest.approx(132.5)
    assert (pkt.mmwave_targets, pkt.mmwave_confidence) == (1, 80)
    assert pkt.presence and pkt.mmwave_present


# ── Decoder behaviour ────────────────────────────────────────────────────────

def test_struct_sizes():
    assert struct.calcsize(VITALS_FMT) == 32
    assert struct.calcsize(FUSED_FMT) == 48


def test_truncated_packets_are_dropped():
    parser = CSIFrameParser()
    assert parser.parse(b"\x01\x00") is None
    assert parser.parse(vitals_packet()[:31]) is None
    assert parser.parse(raw_frame()[:CSI_HEADER_SIZE + 10]) is None   # header claims 128 bytes
    assert parser.n_dropped == 3 and parser.n_parsed == 0


def test_unknown_magic_dropped_known_unhandled_skipped():
    parser = CSIFrameParser()
    assert parser.parse(struct.pack("<I", 0xDEADBEEF) + bytes(40)) is None
    for magic in (CSI_MAGIC_COMPRESSED, CSI_MAGIC_WASM, CSI_MAGIC_ANCHOR):
        assert parser.parse(struct.pack("<I", magic) + bytes(40)) is None
    assert parser.n_dropped == 1 and parser.n_skipped == 3


def test_sequence_gaps_counted_as_loss():
    parser = CSIFrameParser()
    for seq in (10, 11, 15, 16):
        parser.parse(raw_frame(node_id=1, seq=seq))
    st = parser.stats[1]
    assert st.frames == 4 and st.lost == 3
    assert st.loss_ratio == pytest.approx(3 / 7)


def test_sequence_reset_not_counted_as_loss():
    parser = CSIFrameParser()
    parser.parse(raw_frame(seq=500))
    parser.parse(raw_frame(seq=0))     # node rebooted
    assert parser.stats[1].lost == 0


def test_multi_antenna_frame_uses_antenna_zero():
    iq = bytes([0, 3] * 64) + bytes([9, 9] * 64)   # ant0: imag 0, real 3
    frame = CSIFrameParser().parse(raw_frame(n_ant=2, iq=iq))
    assert frame.n_antennas == 2 and frame.n_subcarriers == 64
    np.testing.assert_allclose(frame.amplitude, 3.0)
    np.testing.assert_allclose(frame.phase, 0.0)


def test_expected_node_filter():
    parser = CSIFrameParser(expected_node_id=2)
    assert parser.parse(vitals_packet(node_id=1)) is None
    assert parser.parse(vitals_packet(node_id=2)) is not None


# ── Magic numbers stay unique and in sync with the firmware ──────────────────

def firmware_magics():
    magics = {}
    for header in MAIN_DIR.glob("*.h"):
        for name, value in re.findall(r"#define\s+(\w+_MAGIC)\s+(0x[0-9A-Fa-f]+)", header.read_text()):
            magics[name] = int(value, 16)
    return magics


def test_firmware_packet_magics_unique():
    family = {n: v for n, v in firmware_magics().items() if (v >> 16) == 0xC511}
    values = list(family.values())
    assert len(values) == len(set(values)), f"duplicate packet magic: {family}"


def test_python_magics_match_firmware():
    fw = firmware_magics()
    assert fw["CSI_MAGIC"] == CSI_MAGIC_RAW
    assert fw["EDGE_VITALS_MAGIC"] == CSI_MAGIC_VITALS
    assert fw["EDGE_COMPRESSED_MAGIC"] == CSI_MAGIC_COMPRESSED
    assert fw["EDGE_FUSED_MAGIC"] == CSI_MAGIC_FUSED
    assert fw["WASM_OUTPUT_MAGIC"] == CSI_MAGIC_WASM
