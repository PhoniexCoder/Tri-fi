"""
ml_inference_rf.py — Real-Time Inference with Traditional ML (Random Forest)
A drop-in replacement alternative for ml_inference_v2.py based on Random Forest.

Outputs per frame:
  presence_prob    (float 0–1)     — human presence probability
  breathing_depth  (float 0–1)     — relative tidal-volume proxy
  body_zone        (int  0–3)      — coarse position: 0=absent, 1=near, 2=mid, 3=far
  zone_probs       (float[4])      — per-zone probabilities
"""

import collections
import json
import os
import time
import numpy as np
import joblib

_model_pres_rf  = None
_model_dep_rf   = None
_model_zone_rf  = None
_scaler_rf      = None
_meta_rf        = None

DEFAULT_PREFIX      = "model_rf"
DEFAULT_META_PATH   = "model_rf_meta.json"

# In RF, we predict per-frame, but we can smooth outputs across SEQ_LEN frames.
SEQ_LEN    = 20
N_FEATURES = 27   # 3+10+10+4

PRESENCE_THRESHOLD = 0.60
SMOOTHING_N        = 5

ZONE_LABELS = {0: "absent", 1: "near (<1 m)", 2: "mid (1–3 m)", 3: "far (>3 m)"}


def _load_rf(meta_path):
    global _model_pres_rf, _model_dep_rf, _model_zone_rf, _scaler_rf, _meta_rf
    
    if _model_pres_rf is not None:
        return _model_pres_rf, _model_dep_rf, _model_zone_rf, _scaler_rf, _meta_rf

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing RF meta file at: {meta_path}")

    with open(meta_path) as f:
        _meta_rf = json.load(f)

    # Resolve paths relative to the meta file directory
    base_dir = os.path.dirname(os.path.abspath(meta_path))

    scaler_path = _meta_rf.get("scaler_path")
    pres_path   = _meta_rf.get("model_presence")
    dep_path    = _meta_rf.get("model_depth")
    zone_path   = _meta_rf.get("model_zone")

    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(base_dir, p)

    _scaler_rf     = joblib.load(resolve(scaler_path))
    _model_pres_rf = joblib.load(resolve(pres_path))
    _model_dep_rf  = joblib.load(resolve(dep_path))
    _model_zone_rf = joblib.load(resolve(zone_path))
    
    print(f"[rf] Models loaded successfully ← {meta_path}")
    return _model_pres_rf, _model_dep_rf, _model_zone_rf, _scaler_rf, _meta_rf


class PresenceDetectorRF:
    """
    Random Forest inference engine (similar public API as PresenceDetectorV2).
    """

    def __init__(self,
                 meta_path   = DEFAULT_META_PATH,
                 threshold   = PRESENCE_THRESHOLD,
                 smooth_n    = SMOOTHING_N):

        self.threshold = threshold
        self.smooth_n  = smooth_n

        self._prob_hist    = collections.deque(maxlen=smooth_n)
        self._depth_hist   = collections.deque(maxlen=smooth_n)
        self._zone_probs_h = collections.deque(maxlen=smooth_n)

        # Public state
        self.presence_prob   = 0.0
        self.breathing_depth = 0.0
        self.body_zone       = 0
        self.zone_probs      = np.zeros(4, dtype=np.float32)
        self.smooth_prob     = 0.0
        self.detected        = False
        self.n_frames        = 0

        try:
            _load_rf(meta_path)
            self._mode = "rf"
        except FileNotFoundError:
            print("[rf] default model_rf_meta.json not found — falling back to rules")
            self._mode = "rules"

    def update_from_feature_vector(self, feature_vec: np.ndarray) -> dict:
        self.n_frames += 1
        vec = np.asarray(feature_vec, dtype=np.float32)
        if len(vec) != N_FEATURES:
            return self.status_dict()

        if self._mode == "rf":
            self._infer_rf(vec)
        else:
            self._infer_rules(vec)

        return self.status_dict()

    def update_simple(self, rssi: float, motion: float, breathing: float) -> dict:
        rssi_n  = float(np.clip((rssi + 100) / 70.0, 0.0, 1.0))
        motion  = float(np.clip(motion, 0.0, 1.0))
        breath_n= float(np.clip(breathing, 0.0, 30.0)) / 30.0
        vec     = np.zeros(N_FEATURES, dtype=np.float32)
        vec[0]  = rssi_n
        vec[1]  = motion
        vec[2]  = breath_n
        return self.update_from_feature_vector(vec)

    def is_present(self) -> bool:
        return self.detected

    def zone_label(self) -> str:
        return ZONE_LABELS.get(self.body_zone, "unknown")

    def reset(self):
        self._prob_hist.clear()
        self._depth_hist.clear()
        self._zone_probs_h.clear()
        self.presence_prob   = 0.0
        self.breathing_depth = 0.0
        self.body_zone       = 0
        self.zone_probs      = np.zeros(4, dtype=np.float32)
        self.smooth_prob     = 0.0
        self.detected        = False

    def status_dict(self) -> dict:
        return {
            "ml_prob":           round(self.smooth_prob, 4),
            "ml_detected":       self.detected,
            "ml_breathing_depth":round(self.breathing_depth, 3),
            "ml_body_zone":      self.body_zone,
            "ml_zone_label":     self.zone_label(),
            "ml_zone_probs":     [round(float(p), 3) for p in self.zone_probs],
            "ml_mode":           self._mode,
            "ml_frames":         self.n_frames,
            "ml_window_filled":  self.n_frames >= SEQ_LEN, 
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────────────

    def _infer_rf(self, vec: np.ndarray):
        scaled = _scaler_rf.transform([vec])[0]
        
        # 1. Presence
        pres_prob = _model_pres_rf.predict_proba([scaled])[0][1]
        self._prob_hist.append(pres_prob)
        self.presence_prob = float(pres_prob)
        self.smooth_prob   = float(np.mean(self._prob_hist))
        self.detected      = self.smooth_prob >= self.threshold

        # 2. Depth
        depth_val = _model_dep_rf.predict([scaled])[0]
        self._depth_hist.append(depth_val)
        self.breathing_depth = float(np.mean(self._depth_hist))

        # 3. Zone
        zone_probs_raw = _model_zone_rf.predict_proba([scaled])[0]
        self._zone_probs_h.append(zone_probs_raw)
        self.zone_probs = np.mean(self._zone_probs_h, axis=0) # smooth probabilities
        self.body_zone  = int(np.argmax(self.zone_probs))

    def _infer_rules(self, vec: np.ndarray):
        rssi_n, motion, breath_n = float(vec[0]), float(vec[1]), float(vec[2])
        score = 0.0
        if rssi_n > 0.28:   score += 0.4    # rssi > -80 dBm
        if motion  > 0.05:  score += 0.3
        if breath_n > 0.06: score += 0.3    # breathing > ~2 bpm
        self._prob_hist.append(min(score, 1.0))
        self.smooth_prob = float(np.mean(self._prob_hist))
        self.detected    = self.smooth_prob >= self.threshold
        self.presence_prob = self.smooth_prob

# ─────────────────────────────────────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== PresenceDetectorRF smoke-test ===\n")
    det = PresenceDetectorRF()

    print("Simulating 30 frames — human present, near range")
    for i in range(30):
        vec = np.zeros(N_FEATURES, dtype=np.float32)
        vec[0] = np.clip((np.random.normal(-58, 3) + 100) / 70, 0, 1)  # rssi
        vec[1] = np.random.uniform(0.07, 0.2)                           # motion
        vec[2] = np.random.normal(14, 2) / 30.0                         # breathing
        vec[3] = np.random.normal(0.8, 0.1)    # amp_pca_0
        vec[13]= np.random.normal(-0.5, 0.1)   # phase_pca_0
        vec[23]= 0.6                            # amp_var_mean

        result = det.update_from_feature_vector(vec)
        if (i + 1) % 5 == 0:
            print(f"  frame {i+1:02d} | prob={result['ml_prob']:.3f} | "
                  f"depth={result['ml_breathing_depth']:.3f} | "
                  f"zone={result['ml_zone_label']} | "
                  f"detected={result['ml_detected']}")

    print("\nSmoke-test complete. ✓")
