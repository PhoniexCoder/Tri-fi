"""
Deterministic synthetic session for replay tests (no hardware, no torch).

Three receivers stream raw CSI at 100 Hz for 40 s. From t=20 s a 0.25 Hz
"breathing" modulation appears on nodes 1 and 2. Node 3 loses ~2% of
frames. Vitals arrive once a second, a rescuer badge is active 30-35 s and
confirms at 33 s, and the operator switches to mode 2 at 36 s.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from trifi.config import Config
from trifi.events import Recorder
from trifi.protocol import CSI_MAGIC_RAW, CSI_MAGIC_VITALS, RAW_HEADER_FMT, VITALS_FMT

T0 = 1_700_000_000.0
RATE_HZ = 100
N_SUB = 64
BREATH_START_S = 20.0
BREATH_HZ = 0.25


def raw_packet(node_id: int, seq: int, amplitude: np.ndarray, rssi: int) -> bytes:
    real = np.clip(np.round(amplitude), -127, 127).astype(np.int8)
    iq = np.zeros(2 * len(real), dtype=np.int8)
    iq[1::2] = real                          # ESP-IDF order: (imag, real)
    header = struct.pack(RAW_HEADER_FMT, CSI_MAGIC_RAW, node_id, 1, len(real),
                         2437, seq, rssi, -92, 0)
    return header + iq.tobytes()


def vitals_packet(node_id: int, breathing_bpm: float, motion: float, rssi: int) -> bytes:
    return struct.pack(VITALS_FMT, CSI_MAGIC_VITALS, node_id, 0x01 if breathing_bpm else 0,
                       int(breathing_bpm * 100), 0, rssi, 1 if breathing_bpm else 0,
                       motion, 0.5, 0, 0)


def write_session(path: Path, cfg: Config, seconds: float = 40.0, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    k = np.arange(N_SUB)
    base = {nid: 20 + 10 * np.sin(k / 5 + nid) for nid in (1, 2, 3)}
    rssi = {1: -48, 2: -55, 3: -61}
    seq = {1: 0, 2: 0, 3: 0}

    rec = Recorder(path, cfg)
    n_steps = int(seconds * RATE_HZ)
    mode_switched = confirmed = False
    for i in range(n_steps):
        ts = i / RATE_HZ
        for nid in (1, 2, 3):
            t = T0 + ts + nid * 0.001
            seq[nid] += 1
            if nid == 3 and rng.random() < 0.02:
                continue                       # lost frame (sequence gap)
            amp = base[nid] + rng.normal(0, 0.5, N_SUB)
            if nid in (1, 2) and ts >= BREATH_START_S:
                amp = amp + 8.0 * np.sin(2 * np.pi * BREATH_HZ * ts)
            rec.packet(t, f"192.168.137.{10 + nid}:5000", raw_packet(nid, seq[nid], amp, rssi[nid]))
            if i % RATE_HZ == 0:
                br = 15.0 if (nid == 1 and ts >= BREATH_START_S) else 0.0
                rec.packet(t + 0.0005, f"192.168.137.{10 + nid}:5000",
                           vitals_packet(nid, br, 0.2, rssi[nid]))
        if 30.0 <= ts < 35.0 and i % (RATE_HZ // 5) == 0:
            msg = b"RESCUE_CONFIRMED" if ts >= 33.0 and not confirmed else b"RESCUER_ACTIVE"
            confirmed = confirmed or msg == b"RESCUE_CONFIRMED"
            rec.packet(T0 + ts + 0.004, "192.168.137.50:6000", msg)
        if ts >= 36.0 and not mode_switched:
            rec.mode(T0 + ts + 0.005, 2)
            mode_switched = True
    rec.close()
    return path


def stub_predict(window: np.ndarray) -> np.ndarray:
    """Deterministic stand-in for the ResNet: temporal variation -> presence."""
    temporal = float(np.mean(np.std(window, axis=1)))
    p = float(np.clip((temporal - 0.08) / 0.05, 0.0, 1.0))
    return np.array([1.0 - p, p], dtype=np.float32)
