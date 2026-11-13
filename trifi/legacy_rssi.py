"""
Legacy RSSI-based presence heuristic ("Triangular" mode, op modes 1 and 2).

Ported unchanged in behaviour from examples/ml_inference_triangular.py, except:
  - time comes from the caller (event time), so replays are deterministic;
  - the scikit-learn pickles it loaded but never used are no longer loaded;
  - reset() no longer drops the 'breathing' key.

Known issue (Phase 4): treating an RSSI rise as a sign of presence is
physically doubtful; this detector is kept only for comparison.
"""

from __future__ import annotations

import collections
from collections.abc import Iterable

import numpy as np

EMA_ALPHA = 0.10
DETECTION_HOLD_SECS = 3.5
BASELINE_SAMPLES = 60
WARM_UP_SECS = 8.0
CONSENSUS_DIVISOR = 6.5
NODE_TIMEOUT_SECS = 5.0


class TriangularPresenceDetector:
    def __init__(self, node_ids: Iterable[int], start_t: float):
        self._ids = tuple(node_ids)
        self._nodes = {i: self._blank() for i in self._ids}
        self._baselines: dict[int, collections.deque[float]] = {
            i: collections.deque(maxlen=BASELINE_SAMPLES) for i in self._ids}
        self._baseline_vals = {i: -90.0 for i in self._ids}
        self._start_t = start_t
        self.smooth_prob = 0.0
        self.detected = False
        self._last_present_t = 0.0

    @staticmethod
    def _blank() -> dict:
        return {"rssi": -95.0, "motion": 0.0, "breathing": 0.0, "ts": 0.0}

    def update_node(self, node_id: int, rssi: float, motion: float, breathing: float, t: float) -> None:
        if node_id not in self._nodes:
            return
        self._nodes[node_id] = {"rssi": float(rssi), "motion": float(motion),
                                "breathing": float(breathing), "ts": t}
        # Learn the empty-room baseline only while nobody is detected.
        if not self.detected:
            self._baselines[node_id].append(float(rssi))
            if len(self._baselines[node_id]) >= 15:
                self._baseline_vals[node_id] = float(np.median(self._baselines[node_id]))

    def compute(self, t: float) -> dict:
        active_ids = [i for i in self._ids if t - self._nodes[i]["ts"] < NODE_TIMEOUT_SECS]

        if t - self._start_t < WARM_UP_SECS:
            return self._status(active_ids, 0.0, False)

        # "Strict triple": the weakest node must also see an RSSI jump.
        deltas = sorted((max(0.0, self._nodes[i]["rssi"] - self._baseline_vals[i]) for i in self._ids),
                        reverse=True)
        mean_delta = sum(deltas) / len(deltas)
        strict_score = deltas[-1] * 0.6 + mean_delta * 0.4
        rssi_prob = float(np.clip(strict_score / CONSENSUS_DIVISOR, 0, 1))

        life_boost = 0.0
        for i in self._ids:
            if 8.0 < self._nodes[i]["breathing"] < 35.0:
                life_boost = max(life_boost, 0.4)
            if self._nodes[i]["motion"] > 20.0:
                life_boost = max(life_boost, 0.3)

        total_prob = float(np.clip(rssi_prob + life_boost, 0, 1))
        self.smooth_prob = EMA_ALPHA * total_prob + (1 - EMA_ALPHA) * self.smooth_prob

        if self.smooth_prob > 0.65:
            self._last_present_t = t
            self.detected = True
        elif t - self._last_present_t > DETECTION_HOLD_SECS:
            self.detected = False

        return self._status(active_ids, self.smooth_prob, self.detected)

    def _status(self, active_ids: list[int], prob: float, detected: bool) -> dict:
        return {
            "ml_detected": detected,
            "ml_prob": round(prob, 4),
            "active_nodes": len(active_ids),
            "active_node_ids": active_ids,
            "node_rssi": {i: round(self._nodes[i]["rssi"], 1) for i in self._ids},
            "node_motion": {i: round(self._nodes[i]["motion"], 3) for i in self._ids},
            "node_breathing": {i: round(self._nodes[i]["breathing"], 1) for i in self._ids},
        }

    def reset(self, t: float) -> None:
        self._start_t = t
        self.detected = False
        self.smooth_prob = 0.0
        for i in self._ids:
            self._nodes[i] = self._blank()
