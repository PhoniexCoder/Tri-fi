"""Breathing-rhythm detector: finds 6-30/min rhythms in noisy CSI, ignores drift, sway and noise."""

import numpy as np
import pytest

from trifi.features import MIN_BREATHING_SECONDS, detect_breathing_fft, get_breathing_info


def csi(seconds=30.0, fs=100.0, bpm=None, amp=0.2, noise=1.0, seed=0, n_sub=64, extra=None):
    """(n_sub, n) amplitudes: per-subcarrier white noise plus an optional common rhythm."""
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    x = 20 + rng.normal(0, noise, (n_sub, n))
    if bpm:
        x += amp * np.sin(2 * np.pi * bpm / 60 * t)[None, :]
    if extra is not None:
        x += extra(t)[None, :]
    return x


@pytest.mark.parametrize("bpm,fs", [(15, 100.0), (8, 100.0), (28, 100.0), (12, 20.0), (20, 33.0)])
def test_detects_breathing_and_its_rate(bpm, fs):
    info = get_breathing_info(csi(fs=fs, bpm=bpm), sample_rate=fs)
    assert info["detected"]
    assert info["bpm"] == pytest.approx(bpm, abs=1.0)


def test_regression_old_moving_average_removed_the_band():
    # The old 10-sample detrend high-passed at several Hz, so a clear 100 Hz
    # breathing signal was never detected.
    assert detect_breathing_fft(csi(bpm=15, amp=0.5), 100.0)["detected"]


def test_noise_alone_rarely_triggers():
    hits = sum(detect_breathing_fft(csi(seed=s), 100.0)["detected"] for s in range(40))
    assert hits <= 2


@pytest.mark.parametrize("name,extra", [
    ("linear drift", lambda t: 0.2 * t),
    ("slow wander", lambda t: 3.0 * np.sin(2 * np.pi * 0.03 * t)),
    ("body sway 1 Hz", lambda t: 1.0 * np.sin(2 * np.pi * 1.0 * t)),
])
def test_drift_and_motion_are_not_breathing(name, extra):
    assert not detect_breathing_fft(csi(extra=extra), 100.0)["detected"], name


def test_breathing_survives_drift():
    info = get_breathing_info(csi(bpm=15, amp=0.4, extra=lambda t: 0.2 * t), 100.0)
    assert info["detected"] and info["bpm"] == pytest.approx(15, abs=1.0)


def test_short_buffers_are_not_analysed():
    info = detect_breathing_fft(csi(seconds=MIN_BREATHING_SECONDS - 1, bpm=15, amp=1.0), 100.0)
    assert not info["detected"] and info["dominant_freq"] == 0.0


def test_flat_input_is_safe():
    assert not detect_breathing_fft(np.full((64, 3000), 20.0), 100.0)["detected"]
