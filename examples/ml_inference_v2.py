"""
ml_inference_v2.py — Upgraded Real-Time Multi-Output Inference
Replaces ml_inference.py when model_v2.h5 is available.

Outputs per frame:
  presence_prob    (float 0–1)     — human presence probability
  breathing_depth  (float 0–1)     — relative tidal-volume proxy
  body_zone        (int  0–3)      — coarse position: 0=absent, 1=near, 2=mid, 3=far
  zone_probs       (float[4])      — per-zone softmax probabilities

Graceful fallback: if model_v2.h5 is absent it falls back to ml_inference.py
(v1 model) or to rule-based heuristics.

Usage:
    from ml_inference_v2 import PresenceDetectorV2

    det = PresenceDetectorV2()

    # Call once per CSI frame (must also call extractor.update() first):
    result = det.update_from_feature_vector(feature_vec)

    # Or call with raw scalars (uses v1-compatible 3-feature path):
    result = det.update_simple(rssi, motion, breathing)

    # Read results:
    det.presence_prob     # float
    det.breathing_depth   # float
    det.body_zone         # int (0–3)
    det.is_present()      # bool
"""

import collections
import json
import os
import time
import numpy as np
import joblib

# ── Optional: import v1 as fallback ──────────────────────────────────────────
try:
    from ml_inference import PresenceDetector as _V1Detector
    _HAS_V1 = True
except ImportError:
    _HAS_V1 = False

# ── TF loaded lazily ─────────────────────────────────────────────────────────
_model_v2  = None
_scaler_v2 = None
_meta_v2   = None

DEFAULT_MODEL_PATH  = "model_v2.h5"
DEFAULT_SCALER_PATH = "scaler_v2.pkl"
DEFAULT_META_PATH   = "model_v2_meta.json"

SEQ_LEN    = 20
N_FEATURES = 27   # 3+10+10+4

PRESENCE_THRESHOLD = 0.60
SMOOTHING_N        = 5

ZONE_LABELS = {0: "absent", 1: "near (<1 m)", 2: "mid (1–3 m)", 3: "far (>3 m)"}


def _load_v2(model_path, scaler_path, meta_path):
    global _model_v2, _scaler_v2, _meta_v2
    if _model_v2 is not None:
        return _model_v2, _scaler_v2, _meta_v2

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            _meta_v2 = json.load(f)
        model_path  = _meta_v2.get("model_path",  model_path)
        scaler_path = _meta_v2.get("scaler_path", scaler_path)
    else:
        _meta_v2 = {}

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    import tensorflow as tf
    _model_v2  = tf.keras.models.load_model(model_path, compile=False)
    _scaler_v2 = joblib.load(scaler_path)
    print(f"[v2] Model loaded ← {model_path}")
    return _model_v2, _scaler_v2, _meta_v2


class PresenceDetectorV2:
    """
    Multi-output LSTM inference engine.

    Accepts either:
      A) 27-element feature vectors from CSIFeatureExtractor (full resolution)
      B) 3-element (rssi, motion, breathing) scalars (v1 compatibility path)

    Public state after each update():
      presence_prob    float [0,1]
      breathing_depth  float [0,1]
      body_zone        int   [0,3]
      zone_probs       ndarray (4,)
      smooth_prob      float [0,1]  — moving-averaged presence
      detected         bool
    """

    def __init__(self,
                 model_path  = DEFAULT_MODEL_PATH,
                 scaler_path = DEFAULT_SCALER_PATH,
                 meta_path   = DEFAULT_META_PATH,
                 threshold   = PRESENCE_THRESHOLD,
                 smooth_n    = SMOOTHING_N):

        self.threshold = threshold
        self.smooth_n  = smooth_n

        # Window of feature vectors (27-dim or 3-dim depending on path)
        self._window_v2  = collections.deque(maxlen=SEQ_LEN)
        self._window_v1  = collections.deque(maxlen=SEQ_LEN)
        self._prob_hist  = collections.deque(maxlen=smooth_n)

        # ── Public state ──────────────────────────────────────────────────────
        self.presence_prob   = 0.0
        self.breathing_depth = 0.0
        self.body_zone       = 0
        self.zone_probs      = np.zeros(4, dtype=np.float32)
        self.smooth_prob     = 0.0
        self.detected        = False
        self.n_frames        = 0

        # ── Try loading v2 model ──────────────────────────────────────────────
        try:
            self._model, self._scaler, _ = _load_v2(
                model_path, scaler_path, meta_path)
            self._mode = "v2"
        except FileNotFoundError:
            print("[v2] model_v2.h5 not found — trying v1 fallback")
            if _HAS_V1:
                self._v1 = _V1Detector()
                self._mode = "v1"
            else:
                self._mode = "rules"
            self._model  = None
            self._scaler = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def update_from_feature_vector(self, feature_vec: np.ndarray) -> dict:
        """
        Full-resolution path: receives a (27,) feature vector from
        CSIFeatureExtractor.update().

        Returns the status dict.
        """
        self.n_frames += 1
        vec = np.asarray(feature_vec, dtype=np.float32)
        if len(vec) != N_FEATURES:
            return self.status_dict()   # silently drop malformed

        self._window_v2.append(vec)

        if len(self._window_v2) < SEQ_LEN:
            return self.status_dict()

        if self._mode == "v2":
            self._infer_v2()
        elif self._mode == "v1":
            # v1 only uses first 3 features
            rssi_n, motion, breath_n = float(vec[0]), float(vec[1]), float(vec[2])
            rssi      = rssi_n * 70.0 - 100.0
            breathing = breath_n * 30.0
            self.presence_prob = self._v1.update(rssi, motion, breathing)
            self.smooth_prob   = self.presence_prob
            self.detected      = self.smooth_prob >= self.threshold
        else:
            self._infer_rules_from_vec(vec)

        return self.status_dict()

    def update_simple(self, rssi: float, motion: float, breathing: float) -> dict:
        """
        Compatibility path: 3 scalar inputs (same API as v1 PresenceDetector).
        Builds a padded 27-dim vector (PCA dims = 0) and calls update_from_feature_vector.
        """
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
        self._window_v2.clear()
        self._window_v1.clear()
        self._prob_hist.clear()
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
            "ml_window_filled":  len(self._window_v2) >= SEQ_LEN,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────────────

    def _infer_v2(self):
        """Run one forward pass through the v2 multi-output model."""
        window_arr = np.array(self._window_v2, dtype=np.float32)   # (20, 27)
        normed     = self._scaler.transform(window_arr)              # (20, 27)
        batch      = normed[np.newaxis, ...]                         # (1, 20, 27)

        presence_out, depth_out, zone_out = self._model.predict(
            batch, verbose=0)

        raw_prob = float(presence_out[0, 0])
        self._prob_hist.append(raw_prob)
        self.smooth_prob     = float(np.mean(self._prob_hist))
        self.detected        = self.smooth_prob >= self.threshold
        self.presence_prob   = raw_prob
        self.breathing_depth = float(depth_out[0, 0])
        self.zone_probs      = zone_out[0].astype(np.float32)
        self.body_zone       = int(np.argmax(self.zone_probs))

    def _infer_rules_from_vec(self, vec: np.ndarray):
        """Pure rule-based fallback from normalised feature vector."""
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
    print("=== PresenceDetectorV2 smoke-test ===\n")
    det = PresenceDetectorV2()

    print("Simulating 30 frames — human present, near range")
    for i in range(30):
        vec = np.zeros(N_FEATURES, dtype=np.float32)
        vec[0] = np.clip((np.random.normal(-58, 3) + 100) / 70, 0, 1)  # rssi
        vec[1] = np.random.uniform(0.07, 0.2)                           # motion
        vec[2] = np.random.normal(14, 2) / 30.0                         # breathing
        # Simulate non-zero PCA amplitudes (near zone → high amp_pca_0)
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
