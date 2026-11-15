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
    from trifi.features import preprocess_batch

Usage (real-time ESP32):
    from trifi.features import preprocess_realtime_csi, get_breathing_info

Breathing detection needs a long window (MIN_BREATHING_SECONDS); on short
training windows the breathing features are zero.
"""

from typing import Optional

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

BREATHING_FREQ_MIN = 0.1   # Hz (6 breaths/min)
BREATHING_FREQ_MAX = 0.5   # Hz (30 breaths/min)
ASSUMED_SAMPLE_RATE = 33.0 # Hz (ESP32 CSI typical rate)
MIN_BREATHING_SECONDS = 15.0   # shortest buffer analysed (1.5 cycles at 6/min)
ANALYSIS_RATE_HZ = 10.0        # signal is block-averaged down to about this rate
ANALYSIS_BAND = (0.05, 2.0)    # Hz; spectrum range used for whitening and noise
DRIFT_WINDOW_S = 10.0          # moving average removed as slow drift
BREATHING_SNR_MIN = 12.0       # whitened peak / median noise; simulation-chosen

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
    import h5py  # only needed for batch training data

    with h5py.File(filepath, "r") as f:
        data = f["CSI_amps"][:]
    return data.squeeze(-1).astype(np.float32)


# ── Breathing Detection (FFT) ────────────────────────────────────────────────

def detect_breathing_fft(
    amplitudes: np.ndarray,
    sample_rate: float = ASSUMED_SAMPLE_RATE,
) -> dict:
    """
    Detect a breathing rhythm (0.1-0.5 Hz, 6-30 per minute) in CSI amplitudes.

    Steps: average the most variable quarter of subcarriers; downsample to
    ~10 Hz by block averaging; remove the linear trend and a 10 s moving
    average (slow drift); Hann-window and FFT; whiten the spectrum with a
    power-law fit over 0.05-2 Hz so 1/f drift can't pose as a peak. Breathing
    is reported when the strongest whitened peak in the breathing band is
    BREATHING_SNR_MIN times the median of the rest of the analysis range and
    is also the strongest peak outside the band (rules out body sway and
    other faster motion). Needs at least MIN_BREATHING_SECONDS of data.

    The threshold was chosen in simulation (about 2% false alarms on
    noise-only input, none on 0.6-1.5 Hz sway); calibrate it on the test rig.

    Args:
        amplitudes: shape (n_subcarriers, n_timesteps)
        sample_rate: sampling rate in Hz

    Returns:
        dict with dominant_freq (Hz), band_power (share of 0.05-2 Hz power in
        the breathing band), spectral_entropy (bits; lower = more periodic),
        peak_snr (whitened, capped at 100) and detected.
    """
    n_sub, n_time = amplitudes.shape
    if sample_rate <= 0 or n_time < 30 or n_time / sample_rate < MIN_BREATHING_SECONDS:
        return _empty_breathing()

    # Most variable subcarriers carry the body's modulation; averaging them cuts noise.
    sc_var = np.var(amplitudes, axis=1)
    top_idx = np.argsort(sc_var)[-max(1, n_sub // 4):]
    signal = np.mean(amplitudes[top_idx], axis=0, dtype=np.float64)

    # Block-average down to ~10 Hz: plenty for a 0.5 Hz band, and cheap.
    factor = max(1, int(sample_rate // ANALYSIS_RATE_HZ))
    usable = (len(signal) // factor) * factor
    signal = signal[:usable].reshape(-1, factor).mean(axis=1)
    fs = sample_rate / factor

    # Remove linear trend, then drift slower than ~0.1 Hz (10 s moving average).
    t = np.arange(len(signal))
    signal = signal - np.polyval(np.polyfit(t, signal, 1), t)
    k = int(round(DRIFT_WINDOW_S * fs))
    if 1 < k < len(signal):
        padded = np.pad(signal, (k // 2, k - 1 - k // 2), mode="reflect")
        signal = signal - np.convolve(padded, np.ones(k) / k, mode="valid")

    n_fft = 1 << int(np.ceil(np.log2(len(signal) * 4)))     # zero-pad for a finer peak
    power = np.abs(np.fft.rfft(signal * np.hanning(len(signal)), n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    analysis = (freqs >= ANALYSIS_BAND[0]) & (freqs <= min(ANALYSIS_BAND[1], fs / 2))
    band = (freqs >= BREATHING_FREQ_MIN) & (freqs <= BREATHING_FREQ_MAX)
    other = analysis & ~band
    if not np.any(band) or np.count_nonzero(other) < 4 or np.sum(power[analysis]) < 1e-12:
        return _empty_breathing()

    # Whiten: divide by a power-law fit so drift (1/f) doesn't dominate the band.
    slope, offset = np.polyfit(np.log(freqs[analysis]), np.log(power[analysis] + 1e-20), 1)
    baseline = np.exp(offset + slope * np.log(np.maximum(freqs, 1e-3)))
    white = power / baseline

    band_white = np.where(band, white, 0.0)
    peak = int(np.argmax(band_white))
    noise = float(np.median(white[other]))
    peak_snr = float(white[peak] / (noise + 1e-20))
    dominant = white[peak] >= float(np.max(white[other]))

    p = power[analysis] / np.sum(power[analysis])
    p = p[p > 0]
    return {
        "dominant_freq": float(freqs[peak]),
        "band_power": float(np.sum(power[band & analysis]) / np.sum(power[analysis])),
        "spectral_entropy": float(-np.sum(p * np.log2(p))),
        "peak_snr": min(peak_snr, 100.0),
        "detected": bool(dominant and peak_snr >= BREATHING_SNR_MIN),
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
    file_paths: list[str],
    labels: list[int],
    window_size: int = 100,
    stride: Optional[int] = None,
    sample_rate: float = ASSUMED_SAMPLE_RATE,
    smoothing: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
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

    for i, (fp, label) in enumerate(zip(file_paths, labels, strict=False)):
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
