"""
CSI Preprocessing & Breathing Detection Module
===============================================
Complete feature extraction pipeline for WiFi CSI human presence detection.
Supports both batch training (H5 files) and real-time ESP32 inference.

Features extracted per time window:
  - Statistical: variance, energy, mean, std, max, range
  - Temporal: diff energy, zero-crossing rate, autocorrelation
  - Spatial: subcarrier correlation, variance of means
  - Breathing (FFT): dominant frequency, breathing band power, spectral entropy,
    peak-to-noise ratio, breathing detected flag

Usage (training):
    from csi_preprocessing import preprocess_batch

Usage (real-time ESP32):
    from csi_preprocessing import preprocess_realtime_csi
"""

import numpy as np
import h5py
from typing import Optional, Tuple, List


# ── Constants ─────────────────────────────────────────────────────────────────

BREATHING_FREQ_MIN = 0.1   # Hz (6 breaths/min)
BREATHING_FREQ_MAX = 0.5   # Hz (30 breaths/min)
ASSUMED_SAMPLE_RATE = 33.0 # Hz (ESP32 CSI typical rate)

FEATURE_NAMES = [
    # Statistical features (0-5)
    "mean_variance",
    "mean_energy",
    "mean_amplitude",
    "mean_std",
    "mean_max",
    "mean_range",
    # Temporal features (6-9)
    "mean_diff_energy",
    "zero_crossing_rate",
    "autocorr_lag1",
    "temporal_stability",
    # Spatial features (10-12)
    "global_max",
    "global_energy",
    "variance_of_means",
    # Breathing / FFT features (13-17)
    "breathing_dominant_freq",
    "breathing_band_power",
    "breathing_spectral_entropy",
    "breathing_peak_snr",
    "breathing_detected",
]

N_FEATURES = len(FEATURE_NAMES)


# ── H5 File Loading ──────────────────────────────────────────────────────────

def load_h5_amplitudes(filepath: str) -> np.ndarray:
    """Load CSI amplitude matrix from H5 file. Returns (n_subcarriers, n_timesteps)."""
    with h5py.File(filepath, "r") as f:
        data = f["CSI_amps"][:]
    return data.squeeze(-1).astype(np.float32)


# ── Breathing Detection (FFT) ────────────────────────────────────────────────

def detect_breathing_fft(
    amplitudes: np.ndarray,
    sample_rate: float = ASSUMED_SAMPLE_RATE,
) -> dict:
    """
    Detect breathing signal from CSI amplitude time-series using FFT.

    Analyzes the mean amplitude across subcarriers for periodic patterns
    in the breathing frequency range (0.1-0.5 Hz = 6-30 breaths/min).

    Args:
        amplitudes: shape (n_subcarriers, n_timesteps)
        sample_rate: sampling rate in Hz

    Returns:
        dict with breathing features:
          - dominant_freq: strongest frequency in breathing band (Hz)
          - band_power: normalized power in breathing band
          - spectral_entropy: entropy of power spectrum (lower = more periodic)
          - peak_snr: peak signal-to-noise ratio in breathing band
          - detected: bool, True if breathing pattern detected
    """
    n_sub, n_time = amplitudes.shape

    if n_time < 30:
        return _empty_breathing()

    # Average amplitude across top-variance subcarriers (most sensitive)
    sc_var = np.var(amplitudes, axis=1)
    top_k = max(1, n_sub // 4)
    top_idx = np.argsort(sc_var)[-top_k:]
    signal = np.mean(amplitudes[top_idx], axis=0)

    # Remove DC component and trend
    signal = signal - np.mean(signal)
    # Simple detrending with moving average
    if len(signal) > 10:
        kernel = np.ones(10) / 10
        trend = np.convolve(signal, kernel, mode="same")
        signal = signal - trend

    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(len(signal))
    windowed = signal * window

    # FFT
    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate)
    power = np.abs(fft_vals) ** 2

    # Avoid DC component
    if len(freqs) > 1:
        power[0] = 0

    # Breathing band mask
    breath_mask = (freqs >= BREATHING_FREQ_MIN) & (freqs <= BREATHING_FREQ_MAX)

    if not np.any(breath_mask) or np.sum(power) < 1e-10:
        return _empty_breathing()

    breath_power = power[breath_mask]
    breath_freqs = freqs[breath_mask]
    total_power = np.sum(power[1:])  # exclude DC

    # Dominant frequency in breathing band
    peak_idx = np.argmax(breath_power)
    dominant_freq = float(breath_freqs[peak_idx])

    # Normalized band power (ratio of breathing band to total)
    band_power = float(np.sum(breath_power) / (total_power + 1e-10))

    # Spectral entropy (lower = more periodic/structured)
    p_norm = power[1:] / (np.sum(power[1:]) + 1e-10)
    p_norm = p_norm[p_norm > 0]
    spectral_entropy = float(-np.sum(p_norm * np.log2(p_norm + 1e-15)))

    # Peak SNR: ratio of peak breathing power to mean non-breathing power
    non_breath_power = power[~breath_mask]
    non_breath_power = non_breath_power[1:] if len(non_breath_power) > 1 else non_breath_power
    noise_mean = np.mean(non_breath_power) if len(non_breath_power) > 0 else 1e-10
    peak_snr = float(breath_power[peak_idx] / (noise_mean + 1e-10))

    # Detection: breathing present if band power is significant and SNR is high
    # Field Optimized for ADR-092: 4% band power and 1.1x SNR
    detected = (band_power > 0.04) and (peak_snr > 1.1)

    return {
        "dominant_freq": dominant_freq,
        "band_power": band_power,
        "spectral_entropy": spectral_entropy,
        "peak_snr": min(peak_snr, 100.0),  # cap to avoid extreme values
        "detected": detected,
    }


def _empty_breathing() -> dict:
    return {
        "dominant_freq": 0.0,
        "band_power": 0.0,
        "spectral_entropy": 10.0,
        "peak_snr": 0.0,
        "detected": False,
    }


# ── Feature Extraction ───────────────────────────────────────────────────────

def compute_window_features(
    amplitude_window: np.ndarray,
    sample_rate: float = ASSUMED_SAMPLE_RATE,
) -> np.ndarray:
    """
    Extract all features from a single CSI amplitude window.

    Input: (n_subcarriers, window_length)
    Output: 1D feature vector of length N_FEATURES (18).
    """
    n_sub, n_time = amplitude_window.shape

    # ── Statistical features ──
    sc_var = np.var(amplitude_window, axis=1)
    sc_mean = np.mean(amplitude_window, axis=1)
    sc_std = np.std(amplitude_window, axis=1)
    sc_max = np.max(amplitude_window, axis=1)
    sc_min = np.min(amplitude_window, axis=1)
    sc_range = sc_max - sc_min
    sc_energy = np.mean(amplitude_window ** 2, axis=1)

    # ── Temporal features ──
    diffs = np.diff(amplitude_window, axis=1)
    diff_energy = np.mean(diffs ** 2, axis=1)

    # Zero-crossing rate (averaged across subcarriers)
    mean_signal = np.mean(amplitude_window, axis=0)
    centered = mean_signal - np.mean(mean_signal)
    if len(centered) > 1:
        zcr = float(np.sum(np.abs(np.diff(np.sign(centered))) > 0) / len(centered))
    else:
        zcr = 0.0

    # Autocorrelation at lag 1 (temporal smoothness)
    if n_time > 1:
        autocorr = float(np.corrcoef(mean_signal[:-1], mean_signal[1:])[0, 1])
        if np.isnan(autocorr):
            autocorr = 0.0
    else:
        autocorr = 0.0

    # Temporal stability (inverse of coefficient of variation)
    cv = np.std(mean_signal) / (np.mean(mean_signal) + 1e-10)
    temporal_stability = 1.0 / (1.0 + cv)

    # ── Breathing / FFT features ──
    breath = detect_breathing_fft(amplitude_window, sample_rate)

    # ── Assemble feature vector ──
    features = np.array([
        # Statistical (0-5)
        np.mean(sc_var),
        np.mean(sc_energy),
        np.mean(sc_mean),
        np.mean(sc_std),
        np.mean(sc_max),
        np.mean(sc_range),
        # Temporal (6-9)
        np.mean(diff_energy),
        zcr,
        autocorr,
        temporal_stability,
        # Spatial (10-12)
        np.max(amplitude_window),
        np.sum(sc_energy),
        np.var(sc_mean),
        # Breathing / FFT (13-17)
        breath["dominant_freq"],
        breath["band_power"],
        breath["spectral_entropy"],
        breath["peak_snr"],
        float(breath["detected"]),
    ], dtype=np.float32)

    return features


def extract_window_features(
    amplitudes: np.ndarray,
    window_size: int = 100,
    stride: Optional[int] = None,
    sample_rate: float = ASSUMED_SAMPLE_RATE,
) -> np.ndarray:
    """
    Slide a window across CSI amplitude data and extract features per window.

    Args:
        amplitudes: shape (n_subcarriers, n_timesteps)
        window_size: timesteps per window (100 ≈ 3s at 33Hz)
        stride: step between windows (default = window_size // 2, 50% overlap)
        sample_rate: CSI sampling rate in Hz

    Returns:
        np.ndarray of shape (n_windows, N_FEATURES)
    """
    if stride is None:
        stride = max(1, window_size // 2)

    n_sub, n_time = amplitudes.shape
    features_list = []

    for start in range(0, n_time - window_size + 1, stride):
        window = amplitudes[:, start:start + window_size]
        features_list.append(compute_window_features(window, sample_rate))

    if not features_list:
        features_list.append(compute_window_features(amplitudes, sample_rate))

    return np.array(features_list, dtype=np.float32)


# ── Batch Processing (Training) ──────────────────────────────────────────────

def preprocess_batch(
    file_paths: List[str],
    labels: List[int],
    window_size: int = 100,
    stride: Optional[int] = None,
    sample_rate: float = ASSUMED_SAMPLE_RATE,
    smoothing: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess a batch of H5 files into feature matrix + labels.

    Args:
        file_paths: list of H5 file paths
        labels: list of integer labels
        window_size: timesteps per window
        stride: step between windows
        sample_rate: CSI sampling rate
        smoothing: if True, apply moving average smoothing before extraction

    Returns:
        X: (total_windows, N_FEATURES), y: (total_windows,)
    """
    all_features = []
    all_labels = []
    n_errors = 0

    for i, (fp, label) in enumerate(zip(file_paths, labels)):
        try:
            amps = load_h5_amplitudes(fp)
            if smoothing:
                amps = _smooth_amplitudes(amps)
            feats = extract_window_features(amps, window_size, stride, sample_rate)
            all_features.append(feats)
            all_labels.extend([label] * len(feats))
        except Exception:
            n_errors += 1
            continue

        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{len(file_paths)} files...")

    if not all_features:
        raise ValueError(f"No valid files processed ({n_errors} errors)")

    X = np.vstack(all_features).astype(np.float32)
    y = np.array(all_labels, dtype=np.int32)
    return X, y


def _smooth_amplitudes(amps: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply moving average smoothing along the time axis."""
    if amps.shape[1] <= kernel_size:
        return amps
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.zeros_like(amps)
    for i in range(amps.shape[0]):
        smoothed[i] = np.convolve(amps[i], kernel, mode="same")
    return smoothed


# ── Real-Time Inference (ESP32) ──────────────────────────────────────────────

def preprocess_realtime_csi(
    csi_amplitudes: np.ndarray,
    scaler=None,
    window_size: int = 100,
    sample_rate: float = ASSUMED_SAMPLE_RATE,
    smoothing: bool = True,
) -> np.ndarray:
    """
    Preprocess a CSI amplitude buffer from ESP32 for inference.

    Args:
        csi_amplitudes: (n_subcarriers, n_timesteps) or (n_timesteps,) or (n_sub, n_time, 1)
        scaler: fitted StandardScaler (from scaler.pkl)
        window_size: must match training config
        sample_rate: CSI sampling rate
        smoothing: apply moving average

    Returns:
        np.ndarray of shape (n_windows, N_FEATURES) — ready for model.predict()
    """
    if csi_amplitudes.ndim == 1:
        csi_amplitudes = csi_amplitudes.reshape(1, -1)
    if csi_amplitudes.ndim == 3 and csi_amplitudes.shape[-1] == 1:
        csi_amplitudes = csi_amplitudes.squeeze(-1)
    csi_amplitudes = csi_amplitudes.astype(np.float32)

    if smoothing:
        csi_amplitudes = _smooth_amplitudes(csi_amplitudes)

    features = extract_window_features(csi_amplitudes, window_size, sample_rate=sample_rate)

    if scaler is not None:
        features = scaler.transform(features)

    return features


def get_breathing_info(csi_amplitudes: np.ndarray, sample_rate: float = ASSUMED_SAMPLE_RATE) -> dict:
    """
    Standalone breathing detection for a CSI buffer.
    Returns breathing dict with freq, power, detected flag, BPM estimate.
    """
    if csi_amplitudes.ndim == 1:
        csi_amplitudes = csi_amplitudes.reshape(1, -1)
    if csi_amplitudes.ndim == 3 and csi_amplitudes.shape[-1] == 1:
        csi_amplitudes = csi_amplitudes.squeeze(-1)

    result = detect_breathing_fft(csi_amplitudes, sample_rate)
    result["bpm"] = result["dominant_freq"] * 60.0
    return result
