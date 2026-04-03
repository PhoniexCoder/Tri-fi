"""
csi_feature_extractor.py — Raw CSI → Rich Feature Vector
Converts a window of CSIFrame objects into a (seq_len, n_features) array
suitable for the upgraded LSTM model.

Feature groups produced:
  Group A — 3 original features (backward compatible with old model)
    rssi, motion_energy, breathing_rate

  Group B — 10 PCA components of subcarrier amplitude (presence / breathing depth)
    Captures how the multipath environment changes with body position.

  Group C — 10 PCA components of subcarrier phase-diff (fine motion / breathing phase)
    Phase differences are CFO-free; breathing creates periodic phase oscillation.

  Group D — 4 summary statistics of amplitude variance per frame
    var_mean, var_std, var_max_subcarrier_idx (normalised), snr_estimate

  Total: 3 + 10 + 10 + 4 = 27 features per timestep

The PCA is fit on the first N_PCA_FIT_FRAMES frames seen (online warm-up).
Once fitted it never resets, but can be optionally refreshed.
"""

import numpy as np
import collections
from typing import Optional, Tuple

from csi_frame_parser import CSIFrame

# ── Tunables ──────────────────────────────────────────────────────────────────
N_AMP_COMPONENTS   = 10    # PCA components from amplitude
N_PHASE_COMPONENTS = 10    # PCA components from phase_diff
N_SUMMARY_STATS    = 4     # per-frame scalar summaries
N_ORIGINAL         = 3     # rssi, motion, breathing (from vitals stream)
N_FEATURES_TOTAL   = N_ORIGINAL + N_AMP_COMPONENTS + N_PHASE_COMPONENTS + N_SUMMARY_STATS

N_PCA_FIT_FRAMES   = 200   # frames collected before PCA is fitted
MAX_SUBCARRIERS    = 128   # clip / pad raw arrays to this length


class IncrementalPCA:
    """
    Minimal incremental PCA — no sklearn required at runtime.
    Accumulates frames until n_fit_frames, then fits once via SVD.
    After fitting, projects new frames via the stored components.
    """

    def __init__(self, n_components: int, n_fit_frames: int):
        self.n_components = n_components
        self.n_fit_frames = n_fit_frames
        self._buffer: list = []
        self._components: Optional[np.ndarray] = None  # (n_components, n_features)
        self._mean: Optional[np.ndarray] = None
        self.is_fitted = False

    def partial_fit_and_transform(self, x: np.ndarray) -> np.ndarray:
        """
        x: 1D float array (n_subcarriers,)
        Returns projected vector (n_components,) or zeros if not yet fitted.
        """
        self._buffer.append(x.copy())

        if not self.is_fitted:
            if len(self._buffer) >= self.n_fit_frames:
                self._fit()
            return np.zeros(self.n_components, dtype=np.float32)

        return self._transform(x)

    def _fit(self):
        X = np.array(self._buffer, dtype=np.float64)   # (n_fit, n_sub)
        self._mean = X.mean(axis=0)
        Xc = X - self._mean
        # Economy SVD — only need first n_components right singular vectors
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self._components = Vt[:self.n_components].astype(np.float32)
        self.is_fitted = True
        # Free buffer memory (keep only last 50 for potential re-fit)
        self._buffer = self._buffer[-50:]

    def _transform(self, x: np.ndarray) -> np.ndarray:
        xc = (x - self._mean).astype(np.float32)
        return self._components @ xc   # (n_components,)


def _pad_or_clip(arr: np.ndarray, length: int) -> np.ndarray:
    """Ensure 1-D array has exactly `length` elements."""
    if len(arr) >= length:
        return arr[:length]
    pad = np.zeros(length - len(arr), dtype=arr.dtype)
    return np.concatenate([arr, pad])


class CSIFeatureExtractor:
    """
    Converts raw CSIFrame objects into a rich feature vector per timestep.

    Call update() with each new CSIFrame + current vitals scalars.
    Returns the feature vector once it's ready (PCA warm-up takes
    N_PCA_FIT_FRAMES frames, ~10 s at 20 Hz).

    Typical use inside reader_csi() or SensorHub:
        vector = extractor.update(frame, rssi, motion, breathing)
        if vector is not None:
            sliding_window.append(vector)
    """

    def __init__(self,
                 n_amp_components:   int = N_AMP_COMPONENTS,
                 n_phase_components: int = N_PHASE_COMPONENTS,
                 n_pca_fit_frames:   int = N_PCA_FIT_FRAMES):

        self._pca_amp   = IncrementalPCA(n_amp_components,   n_pca_fit_frames)
        self._pca_phase = IncrementalPCA(n_phase_components, n_pca_fit_frames)

        # Per-subcarrier running variance (Welford) for SNR estimate
        self._var_mean  = np.zeros(MAX_SUBCARRIERS, dtype=np.float64)
        self._var_m2    = np.zeros(MAX_SUBCARRIERS, dtype=np.float64)
        self._var_count = 0

        self.n_features = (N_ORIGINAL + n_amp_components +
                           n_phase_components + N_SUMMARY_STATS)
        self.is_ready   = False   # True once PCA is fitted

    # ──────────────────────────────────────────────────────────────────────────
    # Public
    # ──────────────────────────────────────────────────────────────────────────

    def update(self,
               frame:     CSIFrame,
               rssi:      float,
               motion:    float,
               breathing: float) -> Optional[np.ndarray]:
        """
        Process one CSI frame + associated vitals scalars.

        Returns a float32 feature vector of shape (n_features,),
        or None during PCA warm-up.
        """
        amp = _pad_or_clip(frame.amplitude,  MAX_SUBCARRIERS).astype(np.float32)
        ph  = _pad_or_clip(frame.phase_diff, MAX_SUBCARRIERS - 1).astype(np.float32)
        # Pad phase_diff to MAX_SUBCARRIERS for consistent PCA shape
        ph  = np.concatenate([ph, [0.0]])

        # ── Group D: per-frame amplitude variance stats ───────────────────────
        self._welford_update(amp)
        summary = self._compute_summary(amp)

        # ── Group B: amplitude PCA ────────────────────────────────────────────
        amp_proj = self._pca_amp.partial_fit_and_transform(amp)

        # ── Group C: phase-diff PCA ───────────────────────────────────────────
        ph_proj  = self._pca_phase.partial_fit_and_transform(ph)

        # Mark ready once both PCAs are fitted
        self.is_ready = self._pca_amp.is_fitted and self._pca_phase.is_fitted
        if not self.is_ready:
            return None

        # ── Group A: original 3 scalars ───────────────────────────────────────
        rssi_norm = float(np.clip((rssi + 100) / 70.0, 0.0, 1.0))   # map [-100,-30] → [0,1]
        motion_c  = float(np.clip(motion,    0.0, 1.0))
        breath_n  = float(np.clip(breathing, 0.0, 30.0)) / 30.0

        original = np.array([rssi_norm, motion_c, breath_n], dtype=np.float32)

        # ── Concatenate all groups ────────────────────────────────────────────
        feature = np.concatenate([original, amp_proj, ph_proj, summary])
        return feature.astype(np.float32)

    def feature_names(self):
        """Return human-readable names for each feature index (for debugging)."""
        names = ["rssi_norm", "motion", "breathing_norm"]
        names += [f"amp_pca_{i}" for i in range(self._pca_amp.n_components)]
        names += [f"phase_pca_{i}" for i in range(self._pca_phase.n_components)]
        names += ["amp_var_mean", "amp_var_std", "peak_sub_norm", "snr_db"]
        return names

    # ──────────────────────────────────────────────────────────────────────────
    # Private
    # ──────────────────────────────────────────────────────────────────────────

    def _welford_update(self, amp: np.ndarray):
        """Online per-subcarrier variance via Welford algorithm."""
        self._var_count += 1
        delta = amp - self._var_mean
        self._var_mean += delta / self._var_count
        delta2 = amp - self._var_mean
        self._var_m2   += delta * delta2

    def _compute_summary(self, amp: np.ndarray) -> np.ndarray:
        """
        4 scalar features per frame:
          amp_var_mean    : mean variance across subcarriers (motion energy proxy)
          amp_var_std     : std of variance distribution (body-position spread)
          peak_sub_norm   : which subcarrier has highest current amplitude (0..1)
          snr_db          : signal-to-noise ratio estimate
        """
        # Current-frame variance vs. long-run mean
        var_frame = (amp - self._var_mean) ** 2
        var_mean  = float(var_frame.mean())
        var_std   = float(var_frame.std())

        # Subcarrier with peak amplitude (normalised index 0..1)
        peak_idx  = float(np.argmax(amp)) / MAX_SUBCARRIERS

        # SNR: peak amplitude / noise floor proxy
        noise_est = float(np.percentile(amp, 10)) + 1e-6
        snr_db    = float(20 * np.log10(float(amp.max() + 1e-6) / noise_est))
        snr_norm  = float(np.clip(snr_db / 40.0, 0.0, 1.0))   # 40 dB headroom

        return np.array([var_mean / (var_mean + 1.0),   # soft-normalise
                         np.tanh(var_std),
                         peak_idx,
                         snr_norm], dtype=np.float32)
