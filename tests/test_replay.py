"""
Replay regression: a recorded session must always produce the same output.

The golden file pins today's detection behaviour (including known issues that
Phase 4 will change). When a change is intended, regenerate it with
    TRIFI_UPDATE_GOLDEN=1 python -m pytest tests/test_replay.py
and review the diff.
"""

import json
import os

import pytest
from conftest import TESTS_DIR
from synthetic import T0, stub_predict, write_session

from trifi.config import Config
from trifi.events import read_session, replay

GOLDEN = TESTS_DIR / "data" / "replay_golden.json"


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    return write_session(tmp_path_factory.mktemp("s") / "synthetic.jsonl.gz", Config())


@pytest.fixture(scope="module")
def snapshots(session):
    return list(replay(session, Config(), stub_predict, n_classes=2))


def summarize(snaps):
    return [{
        "t": round(s["t"] - T0, 1),
        "detected": s["ai_detected"],
        "confidence": s["confidence_raw"],
        "status": s["consensus_status"],
        "n": s["n_detecting"],
        "survivors": s["survivors"],
        "br1": s["nodes"][0]["breathing"],
        "rescuer": s["rescuer"]["active"],
        "mode": s["active_op_mode"],
    } for s in snaps]


def test_session_roundtrip(session):
    kinds = [r.kind for r in read_session(session)]
    assert kinds.count("mode") == 1
    assert kinds.count("udp") > 11_000


def test_replay_is_deterministic(session, snapshots):
    again = list(replay(session, Config(), stub_predict, n_classes=2))
    assert summarize(again) == summarize(snapshots)


def test_replay_matches_golden(snapshots):
    summary = summarize(snapshots)
    if os.environ.get("TRIFI_UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(summary, indent=0) + "\n")
    golden = json.loads(GOLDEN.read_text())
    assert len(summary) == len(golden)
    for got, want in zip(summary, golden, strict=True):
        assert got == pytest.approx(want, abs=1e-3), f"diverged at t={want['t']}"


def test_replay_behaviour(snapshots):
    by_t = {round(s["t"] - T0, 1): s for s in snapshots}
    assert len(snapshots) == 200                           # 40 s at 5 Hz (last one at 40.0 s)
    assert all(s["schema"] == 1 for s in snapshots)
    assert not any(by_t[t]["ai_detected"] for t in by_t if 9.0 <= t < 20.0)
    assert all(by_t[t]["ai_detected"] for t in by_t if 25.0 <= t < 40.0)
    # Badge last heard at 34.8 s; it counts as active for 5 s after that.
    assert by_t[32.0]["rescuer"]["active"] and not by_t[40.0]["rescuer"]["active"]
    assert snapshots[-1]["rescuer"]["confirmed"]
    assert by_t[35.0]["active_op_mode"] == 0 and by_t[37.0]["active_op_mode"] == 2
    links = snapshots[-1]["links"]
    assert links["1"]["loss"] == 0.0 and 0.005 < links["3"]["loss"] < 0.04
    assert 95 <= links["1"]["rate_hz"] <= 101
    assert all(c["done"] for c in snapshots[-1]["calibration"].values())


def test_replay_without_model_uses_legacy(session):
    snaps = list(replay(session, Config(), None))
    assert all(not s["model"]["loaded"] for s in snaps)
    assert all(s["csi_ml_prob"] == 0.0 for s in snaps)
