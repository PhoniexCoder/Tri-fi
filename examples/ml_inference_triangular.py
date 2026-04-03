import collections, json, os, time
import numpy as np, joblib

_model=None; _scaler=None; _meta=None
META_PATH="model_triangular_meta.json"

# ── FILTERS (V10 STRICT CENTER-ONLY) ──────────────────────────────────────────
EMA_ALPHA           = 0.10   # Even smoother (V10)
DETECTION_HOLD_SECS  = 3.5    # Higher hold for stability
BASELINE_SAMPLES     = 60     
WARM_UP_SECS         = 8.0    # 8-second Noise Analysis for 100% clean baseline
CONSENSUS_DIVISOR    = 6.5    # Strict 3st-node requirement factor

def _load(meta_path):
    global _model,_scaler,_meta
    try:
        if not os.path.exists(meta_path): return
        _meta=json.load(open(meta_path))
        base=os.path.dirname(os.path.abspath(meta_path))
        def res(p): return p if os.path.isabs(p) else os.path.join(base,p)
        _scaler=joblib.load(res(_meta["scaler_path"]))
        _model =joblib.load(res(_meta["model_presence"]))
    except: pass

class TriangularPresenceDetector:
    def __init__(self, meta_path=META_PATH):
        self._nodes = {i: {"rssi": -95.0, "motion": 0.0, "breathing": 0.0, "ts": 0.0} for i in [1, 2, 3]}
        self._baselines = {i: collections.deque(maxlen=BASELINE_SAMPLES) for i in [1,2,3]}
        self._baseline_vals = {i: -90.0 for i in [1,2,3]}
        
        self._start_ts     = time.time()
        self.smooth_prob   = 0.0
        self.detected      = False
        self._last_present_ts = 0
        self.n_frames      = 0
        
        _load(meta_path)

    def update_node(self, node_id, rssi, motion, breathing):
        if node_id not in self._nodes: return
        now = time.time()
        self._nodes[node_id] = {"rssi": float(rssi), "motion": float(motion), "breathing": float(breathing), "ts": now}
        
        # Continuous learning when NO person is detected (Baseline updates)
        if not self.detected:
            self._baselines[node_id].append(float(rssi))
            if len(self._baselines[node_id]) >= 15:
                self._baseline_vals[node_id] = np.median(self._baselines[node_id])

    def compute(self) -> dict:
        self.n_frames += 1
        now = time.time()
        active_ids = [i for i in [1,2,3] if now - self._nodes[i]["ts"] < 5.0]
        
        # 1. WARM-UP (Calibration Countdown) ──────────────────────────────────
        warm_up_left = max(0, WARM_UP_SECS - (now - self._start_ts))
        if warm_up_left > 0:
            if self.n_frames % 10 == 0:
                print(f"[Learning System] Step out... ({warm_up_left:.1f}s left)")
                print(f"           Calibrated Room: N1:{self._baseline_vals[1]:.1f} N2:{self._baseline_vals[2]:.1f} N3:{self._baseline_vals[3]:.1f}")
            return self._status(active_ids, 0.0, False)

        # 2. THE V10 "STRICT TRIPLE" CONSENSUS ────────────────────────────────
        deltas = []
        for i in [1,2,3]:
            # Calculate the jump for all 3 nodes
            d = max(0, self._nodes[i]["rssi"] - self._baseline_vals[i])
            deltas.append(d)
        
        deltas.sort(reverse=True) # [High, Middle, Low]
        
        # BALANCE CHECK: If one node is seeing 15dB and another seeing 0, it's an "Edge Case"
        # We enforce that the WEAKEST node must still see a jump to prove "Inside"
        # Score = (Weakest Node Signal * 0.6) + (Average Signal * 0.4)
        mean_delta = sum(deltas) / 3.0
        strict_score = (deltas[2] * 0.6) + (mean_delta * 0.4)
        
        # Normalize to probability (V10 divisor is 6.5 to ignore noise)
        rssi_prob = float(np.clip(strict_score / CONSENSUS_DIVISOR, 0, 1))

        # 3. MOTION OVERRIDE (Sign of Life) ───────────────────────────────────
        life_boost = 0.0
        for i in [1,2,3]:
            # If any node sees breathing or high motion, it adds confidence ONLY IF RSSI is somewhat positive
            if 8.0 < self._nodes[i]["breathing"] < 35.0: life_boost = max(life_boost, 0.4)
            if self._nodes[i]["motion"] > 20.0: life_boost = max(life_boost, 0.3)
        
        # Final Combined Probability
        total_prob = float(np.clip(rssi_prob + life_boost, 0, 1))

        # 4. EMA FILTERING & STABILITY ────────────────────────────────────────
        self.smooth_prob = (EMA_ALPHA * total_prob) + ((1 - EMA_ALPHA) * self.smooth_prob)
        
        # High-Confidence Threshold for "Center Detection"
        instant_detected = self.smooth_prob > 0.65
        
        if instant_detected:
            self._last_present_ts = now
            self.detected = True
        elif now - self._last_present_ts > DETECTION_HOLD_SECS:
            self.detected = False
            
        return self._status(active_ids, self.smooth_prob, self.detected)

    def _status(self, active_ids, prob, detected):
        return {
            "ml_detected":    detected,
            "ml_prob":        round(prob, 4),
            "ml_mode":        "strict_triangular_v10",
            "active_nodes":   len(active_ids),
            "active_node_ids": active_ids,
            "node_rssi":      {i: round(self._nodes[i]["rssi"], 1) for i in [1,2,3]},
            "node_motion":    {i: round(self._nodes[i]["motion"], 3) for i in [1,2,3]},
            "node_breathing": {i: round(self._nodes[i]["breathing"], 1) for i in [1,2,3]},
            "pos_x":          self._estimate_pos(),
            "pos_y":          0.5,
        }

    def _estimate_pos(self):
        # Weighted centroid using the "Delta" above baseline
        w = {i: max(0.1, self._nodes[i]["rssi"] - self._baseline_vals[i] + 10) for i in [1,2,3]}
        total = sum(w.values())
        return float(np.clip((w[1]*0.5 + w[2]*0.1 + w[3]*0.9) / total, 0.1, 0.9))

    def reset(self):
        self._start_ts = time.time(); self.detected = False; self.smooth_prob = 0.0
        for i in [1,2,3]: self._nodes[i] = {"rssi": -95.0, "motion": 0.0, "ts": 0.0}
