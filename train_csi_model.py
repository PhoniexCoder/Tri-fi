"""
Train CSI Human Presence Detection Model (Complete System)
==========================================================
Uses CSI-Bench dataset with full feature extraction + FFT breathing detection.
Trains a Random Forest binary classifier: 0 = no_human, 1 = human_present.

Sources:
  - BreathingDetection/empty → label 0 (no human)
  - BreathingDetection/sleep → label 1 (breathing human)
  - HumanActivityRecognition/* → label 1 (moving human)

Outputs:
  - model.pkl        — trained Random Forest
  - scaler.pkl       — StandardScaler
  - model_meta.json  — metadata, accuracy, feature importances

Usage:
    python train_csi_model.py
    python train_csi_model.py --sample 0.10   # use 10% of data
    python train_csi_model.py --sample 0.05   # use 5% (faster)
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from csi_preprocessing import (
    FEATURE_NAMES,
    N_FEATURES,
    preprocess_batch,
)


# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.resolve()
DATASET_ROOT = PROJECT_ROOT / "archive" / "csi-bench-dataset" / "csi-bench-dataset"

DEFAULTS = {
    "sample_fraction": 0.07,
    "window_size": 100,       # ~3 seconds at 33Hz
    "test_split": 0.2,
    "random_seed": 42,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 25,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "n_jobs": -1,
    "random_state": DEFAULTS["random_seed"],
    "class_weight": "balanced",
}


# ── Dataset Discovery ────────────────────────────────────────────────────────

def discover_files(dataset_root: Path) -> dict:
    """Walk CSI-Bench and collect H5 files grouped by binary label."""
    files = {0: [], 1: []}

    # Source 1: BreathingDetection
    breathing_root = dataset_root / "BreathingDetection" / "sub_Human"
    if breathing_root.exists():
        for h5 in breathing_root.rglob("*.h5"):
            path_str = str(h5)
            if "act_empty" in path_str:
                files[0].append(path_str)
            elif "act_sleep" in path_str:
                files[1].append(path_str)

    # Source 2: HumanActivityRecognition (all = human present)
    har_root = dataset_root / "HumanActivityRecognition" / "sub_Human"
    if har_root.exists():
        activity_prefixes = [
            "act_jumping", "act_running", "act_seated-breathing",
            "act_walking-case", "act_wavinghand",
        ]
        for h5 in har_root.rglob("*.h5"):
            path_str = str(h5)
            if any(p in path_str for p in activity_prefixes):
                files[1].append(path_str)

    return files


def sample_files(files: dict, fraction: float, seed: int):
    """Stratified sampling of file paths and labels."""
    rng = random.Random(seed)
    paths, labels = [], []
    for label, file_list in files.items():
        n = max(10, int(len(file_list) * fraction))
        n = min(n, len(file_list))
        selected = rng.sample(file_list, n)
        paths.extend(selected)
        labels.extend([label] * len(selected))
    return paths, labels


# ── Main Training Pipeline ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train CSI Human Detection Model")
    parser.add_argument("--sample", type=float, default=DEFAULTS["sample_fraction"],
                        help="Fraction of dataset to use (0.01-1.0)")
    parser.add_argument("--window", type=int, default=DEFAULTS["window_size"],
                        help="Window size in timesteps")
    args = parser.parse_args()

    # Validate CLI arguments
    if args.sample <= 0 or args.sample > 1:
        parser.error(f"--sample must be > 0 and <= 1, got {args.sample}")
    if args.window <= 0:
        parser.error(f"--window must be a positive integer, got {args.window}")

    sample_frac = args.sample
    window_size = args.window
    seed = DEFAULTS["random_seed"]

    print("=" * 60)
    print("  CSI Human Detection — Training Pipeline")
    print("  Features: Statistical + Temporal + Spatial + Breathing FFT")
    print("  Model: Random Forest (CPU-optimized)")
    print("=" * 60)

    # ── Step 1: Discover ──
    print(f"\n[1/6] Discovering dataset at {DATASET_ROOT}...")
    if not DATASET_ROOT.exists():
        print(f"ERROR: Dataset not found at {DATASET_ROOT}")
        sys.exit(1)

    files = discover_files(DATASET_ROOT)
    print(f"  Class 0 (no_human):      {len(files[0]):>7,} files")
    print(f"  Class 1 (human_present): {len(files[1]):>7,} files")
    print(f"  Total:                   {sum(len(v) for v in files.values()):>7,} files")

    if len(files[0]) == 0 or len(files[1]) == 0:
        print("ERROR: Need files in both classes.")
        sys.exit(1)

    # ── Step 2: Sample ──
    print(f"\n[2/6] Sampling {sample_frac*100:.0f}% of files per class...")
    paths, labels = sample_files(files, sample_frac, seed)
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        name = "no_human" if u == 0 else "human_present"
        print(f"  Class {u} ({name}): {c:,} files")
    print(f"  Total sampled: {len(paths):,} files")

    # ── Step 3: Extract features ──
    print(f"\n[3/6] Extracting {N_FEATURES} features per window (window_size={window_size})...")
    print(f"  Features: {', '.join(FEATURE_NAMES[:6])}...")
    print(f"            + breathing FFT (dominant_freq, band_power, spectral_entropy, peak_snr)")
    t0 = time.time()
    X, y = preprocess_batch(paths, labels, window_size=window_size, smoothing=True)
    t_extract = time.time() - t0
    print(f"  Feature matrix: {X.shape[0]:,} windows × {X.shape[1]} features")
    print(f"  Extraction time: {t_extract:.1f}s")

    # Clean NaN/Inf
    mask = np.isfinite(X).all(axis=1)
    if not mask.all():
        n_bad = (~mask).sum()
        print(f"  Cleaned {n_bad} windows with NaN/Inf")
        X, y = X[mask], y[mask]

    # ── Step 4: Normalize ──
    print("\n[4/6] Normalizing features (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=DEFAULTS["test_split"],
        random_state=seed, stratify=y,
    )
    print(f"  Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # ── Step 5: Train ──
    print(f"\n[5/6] Training Random Forest ({RF_PARAMS['n_estimators']} trees, "
          f"max_depth={RF_PARAMS['max_depth']})...")
    t0 = time.time()
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    t_train = time.time() - t0
    print(f"  Training time: {t_train:.1f}s")

    # ── Step 6: Evaluate ──
    print("\n[6/6] Evaluating...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    target_names = ["no_human", "human_present"]
    report = classification_report(y_test, y_pred, target_names=target_names)

    print(f"\n  *** Accuracy: {acc*100:.2f}% ***\n")
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                    no_human  human_present")
    print(f"  Actual no_human     {cm[0][0]:>6}         {cm[0][1]:>6}")
    print(f"  Actual human        {cm[1][0]:>6}         {cm[1][1]:>6}")

    # Feature importances
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n  Feature Importances (top 10):")
    for rank, i in enumerate(sorted_idx[:10], 1):
        bar = "█" * int(importances[i] * 50)
        print(f"    {rank:>2}. {FEATURE_NAMES[i]:<28s} {importances[i]:.4f} {bar}")

    # Breathing feature analysis
    print(f"\n  Breathing Feature Analysis:")
    breath_features = ["breathing_dominant_freq", "breathing_band_power",
                       "breathing_spectral_entropy", "breathing_peak_snr", "breathing_detected"]
    for name in breath_features:
        if name not in FEATURE_NAMES:
            print(f"    {name:<28s} WARNING: not in FEATURE_NAMES, skipping")
            continue
        idx = FEATURE_NAMES.index(name)
        print(f"    {name:<28s} importance={importances[idx]:.4f}")

    # ── Save ──
    print("\n" + "=" * 60)
    print("  Saving model artifacts...")

    model_path = PROJECT_ROOT / "model.pkl"
    scaler_path = PROJECT_ROOT / "scaler.pkl"
    meta_path = PROJECT_ROOT / "model_meta.json"

    joblib.dump(model, model_path)
    print(f"  model.pkl:      {model_path} ({os.path.getsize(model_path):,} bytes)")

    joblib.dump(scaler, scaler_path)
    print(f"  scaler.pkl:     {scaler_path}")

    meta = {
        "task": "csi_human_presence_detection",
        "classes": {"0": "no_human", "1": "human_present"},
        "model_type": "RandomForestClassifier",
        "n_features": N_FEATURES,
        "feature_names": FEATURE_NAMES,
        "window_size": window_size,
        "sample_rate_hz": 33.0,
        "accuracy": round(float(acc), 4),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "sample_fraction": sample_frac,
        "smoothing": True,
        "breathing_detection": {
            "method": "FFT",
            "freq_range_hz": [0.1, 0.5],
            "features": breath_features,
        },
        "rf_params": {k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
        "feature_importances": {
            FEATURE_NAMES[i]: round(float(importances[i]), 4)
            for i in sorted_idx
        },
        "triangle_consensus": {
            "nodes": 3,
            "min_detect": 2,
            "positions": {"1": [0, 0], "2": [1, 0], "3": [0.5, 1]},
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  model_meta.json: {meta_path}")

    print(f"\n  Total time: {t_extract + t_train:.1f}s")
    print("=" * 60)
    print("  DONE. Model ready for deployment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
