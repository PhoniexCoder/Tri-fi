"""Pipeline unit behaviour (ported logic, event-time driven)."""

import numpy as np
import pytest
from synthetic import raw_packet, vitals_packet

from trifi.config import Config
from trifi.inference import ModelInfo
from trifi.pipeline import CALIB_WINDOWS, InferenceRequest, Pipeline

T = 1000.0


def make(model_classes=2, **cfg):
    model = ModelInfo(path=None, n_classes=model_classes) if model_classes else None
    return Pipeline(Config(**cfg), start_t=T, model=model)


def feed(pipe, node_id, n, t0=T, rate=100.0, amp=None):
    reqs = []
    rng = np.random.default_rng(node_id)
    for i in range(n):
        a = amp if amp is not None else 20 + rng.normal(0, 0.5, 64)
        reqs += pipe.on_packet(raw_packet(node_id, i + 1, a, -50), "10.0.0.1:1", t0 + i / rate)
    return reqs


def test_first_request_after_full_window_then_every_half_window():
    pipe = make()
    reqs = []
    for i in range(250):
        reqs += pipe.on_packet(raw_packet(1, i + 1, np.full(64, 20.0), -50), "a:1", T + i / 100)
        for r in reqs[len(reqs) - 1:] if reqs and pipe.nodes[1].pending else []:
            pipe.on_inference(r, np.array([1.0, 0.0]))
    assert [r.t for r in reqs] == pytest.approx([T + 0.99, T + 1.49, T + 1.99, T + 2.49])
    assert all(r.window.shape == (64, 100) for r in reqs)


def test_pending_request_drops_new_windows():
    pipe = make()
    reqs = feed(pipe, 1, 200)      # never answered
    assert len(reqs) == 1
    assert pipe.nodes[1].dropped_requests == 2


def test_no_model_no_requests_but_calibration_and_breathing_run():
    pipe = make(model_classes=0)
    assert feed(pipe, 1, 900) == []
    node = pipe.nodes[1]
    assert node.noise_std is not None and len(node.calib) == CALIB_WINDOWS
    assert node.breathing["bpm"] >= 0.0


def test_gate_blocks_until_calibrated_unless_confident():
    pipe = make()
    req = feed(pipe, 1, 100)[0]
    assert req.gated                                   # no baseline yet
    pipe.on_inference(req, np.array([0.3, 0.7]))       # not confident -> forced empty
    assert pipe.nodes[1].ml_prob == 0.0
    req2 = InferenceRequest(1, req.window, gated=True, t=T + 2)
    pipe.nodes[1].pending = True
    pipe.on_inference(req2, np.array([0.02, 0.98]))    # > 0.95 overrides the gate
    assert pipe.nodes[1].ml_prob == pytest.approx(0.49, abs=1e-6)


def test_six_class_survivor_mapping():
    pipe = make(model_classes=6)
    req = InferenceRequest(1, np.zeros((64, 100)), gated=False, t=T)
    for _ in range(3):
        pipe.nodes[1].pending = True
        pipe.on_inference(req, np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))   # class 2 = two people
    assert pipe.nodes[1].survivors() == 2
    assert pipe.nodes[1].detected


def test_unknown_nodes_and_garbage_ignored():
    pipe = make()
    assert pipe.on_packet(raw_packet(9, 1, np.full(64, 20.0), -50), "a:1", T) == []
    assert pipe.on_packet(b"\x00" * 50, "a:1", T) == []
    assert pipe.parser.n_dropped == 1


def test_vitals_update_node_state():
    pipe = make()
    pipe.on_packet(vitals_packet(2, 14.5, 0.3, -58), "a:1", T)
    n = pipe.nodes[2]
    assert n.vitals_breathing == pytest.approx(14.5) and n.rssi == -58


def test_rescuer_messages():
    pipe = make()
    pipe.on_packet(b"RESCUER_ACTIVE", "10.0.0.9:7", T)
    assert pipe.snapshot(T + 1)["rescuer"]["active"]
    pipe.on_packet(b"RESCUE_CONFIRMED", "10.0.0.9:7", T + 2)
    s = pipe.snapshot(T + 8)
    assert s["rescuer"]["confirmed"] and not s["rescuer"]["active"]
    assert pipe.rescuer.ip == "10.0.0.9"


def test_set_mode_validates():
    pipe = make()
    pipe.set_mode(1)
    assert pipe.snapshot(T)["active_op_mode"] == 1
    with pytest.raises(ValueError):
        pipe.set_mode(5)


def test_snapshot_schema_and_heatmap_cadence():
    pipe = make()
    snaps = [pipe.snapshot(T + i * 0.2) for i in range(10)]
    required = {"type", "schema", "t", "ai_detected", "ai_prob", "confidence_raw", "nodes",
                "links", "calibration", "model", "rescuer", "survivors_list"}
    assert all(required <= s.keys() for s in snaps)
    assert ["heatmap" in s for s in snaps] == [False] * 9 + [True]
    assert len(snaps[-1]["heatmap"]) == 400


def test_link_rate_and_loss():
    pipe = make()
    for i in range(300):
        if i % 50 != 7:                      # drop 6 of 300 frames
            pipe.on_packet(raw_packet(1, i + 1, np.full(64, 20.0), -50), "a:1", T + i / 100)
    links = pipe.link_stats(T + 2.99)["1"]
    assert links["frames"] == 294
    assert links["loss"] == pytest.approx(6 / 300, abs=1e-3)
    assert 95 <= links["rate_hz"] <= 100
