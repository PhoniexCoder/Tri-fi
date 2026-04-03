"""
CSI Preprocessing Module — Reusable for Training & Real-Time ESP32 Inference

Extracts statistical features from CSI amplitude data across time windows.
Works with any subcarrier count (56, 64, 128, 256, etc.) by producing
fixed-length feature vectors via statistical aggregation.

Usage (training):
    from csi_preprocessing import load_h5_amplitudes, extract_window_features, preprocess_batch

Usage (real-time ESP32):
    from csi_preprocessing import preprocess_realtime_csi
"""

import numpy as np
import h5py
from typing import Optional, Tuple, List


# ── H5 File Loading ──────────────────────────────────────────────────────────

def load_h5_amplitudes(filepath: str) -> np.ndarray:
    """
    Load CSI amplitude matrix from an H5 file.

    Args:
        filepath: Path to .h5 file containing 'CSI_amps' dataset.

    Returns:
        np.ndarray of shape (n_subcarriers, n_timesteps) — float32
    """
    with h5py.File(filepath, "r") as f:
        data = f["CSI_amps"][:]  # shape: (subcarriers, timesteps, 1)
    return data.squeeze(-1).astype(np.float32)  # → (subcarriers, timesteps)


# ── Feature Extraction ───────────────────────────────────────────────────────

def compute_window_features(amplitude_window: np.ndarray) -> np.ndarray:
    """
    Extract a fixed-length feature vector from a single CSI amplitude window.

    Input shape: (n_subcarriers, window_length)
    Output: 1D feature vector of length 10.

    Features (aggregated across subcarriers):
        0: mean_variance       — average per-subcarrier variance
        1: mean_energy         — average signal energy (mean of squared values)
        2: mean_amplitude      — global mean amplitude
        3: mean_std            — average per-subcarrier std deviation
        4: mean_max            — average per-subcarrier max
        5: mean_range          — average per-subcarrier range (max - min)
        6: global_max          — absolute maximum amplitude
        7: global_energy       — total energy across all subcarriers
        8: variance_of_means   — variance of per-subcarrier means (spatial spread)
        9: mean_diff_energy    — average energy of first-order differences (motion sensitivity)
    """
    # Per-subcarrier statistics
    sc_var = np.var(amplitude_window, axis=1)       # (n_subcarriers,)
    sc_mean = np.mean(amplitude_window, axis=1)     # (n_subcarriers,)
    sc_std = np.std(amplitude_window, axis=1)       # (n_subcarriers,)
    sc_max = np.max(amplitude_window, axis=1)       # (n_subcarriers,)
    sc_min = np.min(amplitude_window, axis=1)       # (n_subcarriers,)
    sc_range = sc_max - sc_min                      # (n_subcarriers,)

    # Energy: mean of squared amplitudes per subcarrier
    sc_energy = np.mean(amplitude_window ** 2, axis=1)  # (n_subcarriers,)

    # First-order temporal differences → motion sensitivity
    diffs = np.diff(amplitude_window, axis=1)
    diff_energy = np.mean(diffs ** 2, axis=1)  # (n_subcarriers,)

    features = np.array([
        np.mean(sc_var),            # 0: mean_variance
        np.mean(sc_energy),         # 1: mean_energy
        np.mean(sc_mean),           # 2: mean_amplitude
        np.mean(sc_std),            # 3: mean_std
        np.mean(sc_max),            # 4: mean_max
        np.mean(sc_range),          # 5: mean_range
        np.max(amplitude_window),   # 6: global_max
        np.sum(sc_energy),          # 7: global_energy
        np.var(sc_mean),            # 8: variance_of_means
        np.mean(diff_energy),       # 9: mean_diff_energy
    ], dtype=np.float32)

    return features


def extract_window_features(
    amplitudes: np.ndarray,
    window_size: int = 100,
    stride: Optional[int] = None,
) -> np.ndarray:
    """
    Slide a window across CSI amplitude data and extract features per window.

    Args:
        amplitudes: shape (n_subcarriers, n_timesteps)
        window_size: number of timesteps per window (default 100 ≈ 3s at ~33Hz)
        stride: step between windows (default = window_size, no overlap)

    Returns:
        np.ndarray of shape (n_windows, 10) — one feature vector per window
    """
    if stride is None:
        stride = window_size

    n_subcarriers, n_timesteps = amplitudes.shape
    features_list = []

    for start in range(0, n_timesteps - window_size + 1, stride):
        window = amplitudes[:, start:start + window_size]
        features_list.append(compute_window_features(window))

    if not features_list:
        # File too short for even one window — use entire recording
        features_list.append(compute_window_features(amplitudes))

    return np.array(features_list, dtype=np.float32)


# ── Batch Processing (Training) ──────────────────────────────────────────────

def preprocess_batch(
    file_paths: List[str],
    labels: List[int],
    window_size: int = 100,
    stride: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess a batch of H5 files into feature matrix + labels.

    Args:
        file_paths: list of paths to .h5 files
        labels: list of integer labels (same length as file_paths)
        window_size: timesteps per window
        stride: step between windows

    Returns:
        X: np.ndarray of shape (total_windows, 10)
        y: np.ndarray of shape (total_windows,)
    """
    all_features = []
    all_labels = []

    for fp, label in zip(file_paths, labels):
        try:
            amps = load_h5_amplitudes(fp)
            feats = extract_window_features(amps, window_size, stride)
            all_features.append(feats)
            all_labels.extend([label] * len(feats))
        except Exception as e:
            # Skip corrupted files silently
            continue

    X = np.vstack(all_features).astype(np.float32)
    y = np.array(all_labels, dtype=np.int32)
    return X, y


# ── Real-Time Inference (ESP32) ──────────────────────────────────────────────

def preprocess_realtime_csi(
    csi_amplitudes: np.ndarray,
    scaler=None,
    window_size: int = 100,
) -> np.ndarray:
    """
    Preprocess a single CSI amplitude buffer from an ESP32 node for inference.

    This is the function to call in the real-time pipeline. It accepts raw
    amplitude data, extracts features, and optionally normalizes them.

    Args:
        csi_amplitudes: np.ndarray of shape (n_subcarriers, n_timesteps)
            Raw CSI amplitude values from ESP32. Subcarrier count can be
            any value (64 for ESP32-S3, etc.)
        scaler: fitted sklearn StandardScaler (loaded from scaler.pkl).
            If None, returns unnormalized features.
        window_size: timesteps per window (must match training config).

    Returns:
        np.ndarray of shape (n_windows, 10) — ready for model.predict()

    Example:
        import joblib
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')

        # csi_buffer shape: (64, 100) from ESP32
        features = preprocess_realtime_csi(csi_buffer, scaler)
        prediction = model.predict(features)
        # prediction[0] == 0 → no human, 1 → human present
    """
    if csi_amplitudes.ndim == 1:
        # Single subcarrier stream → reshape to (1, n_timesteps)
        csi_amplitudes = csi_amplitudes.reshape(1, -1)

    if csi_amplitudes.ndim == 3 and csi_amplitudes.shape[-1] == 1:
        csi_amplitudes = csi_amplitudes.squeeze(-1)

    csi_amplitudes = csi_amplitudes.astype(np.float32)

    features = extract_window_features(csi_amplitudes, window_size)

    if scaler is not None:
        features = scaler.transform(features)

    return features


# ── Feature Names (for debugging/analysis) ────────────────────────────────────

FEATURE_NAMES = [
    "mean_variance",
    "mean_energy",
    "mean_amplitude",
    "mean_std",
    "mean_max",
    "mean_range",
    "global_max",
    "global_energy",
    "variance_of_means",
    "mean_diff_energy",
]
