"""
Train a lightweight CSI Human Presence Detection model.

Uses the CSI-Bench dataset from archive/csi-bench-dataset to train a Random Forest
binary classifier: 0 = no_human, 1 = human_present.

Sources:
  - BreathingDetection/empty  → label 0 (no human)
  - BreathingDetection/sleep  → label 1 (human breathing)
  - HumanActivityRecognition/* → label 1 (human movement)

Outputs (saved to project root):
  - model.pkl       — trained Random Forest model
  - scaler.pkl      — fitted StandardScaler for feature normalization
  - model_meta.json — metadata (features, accuracy, config)

Usage:
    python train_csi_model.py
"""

import os
import sys
import json
import time
import glob
import random
import numpy as np
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from csi_preprocessing import (
    preprocess_batch,
    FEATURE_NAMES,
)


# ── Configuration ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.resolve()
DATASET_ROOT = PROJECT_ROOT / "archive" / "csi-bench-dataset" / "csi-bench-dataset"

SAMPLE_FRACTION = 0.07       # 7% of files per class (speed vs quality tradeoff)
WINDOW_SIZE = 100            # ~3 seconds at ~33Hz sampling rate
RANDOM_SEED = 42
TEST_SPLIT = 0.2

# Model hyperparameters (optimized for CPU speed)
RF_PARAMS = {
    "n_estimators": 150,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "n_jobs": -1,             # use all CPU cores
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",  # handle class imbalance
}


# ── Dataset Discovery ────────────────────────────────────────────────────────

def discover_files(dataset_root: Path) -> dict:
    """
    Walk the CSI-Bench dataset and collect H5 file paths grouped by binary label.

    Returns:
        {0: [list of no-human file paths], 1: [list of human-present file paths]}
    """
    files = {0: [], 1: []}

    # ── Source 1: BreathingDetection ──
    breathing_root = dataset_root / "BreathingDetection" / "sub_Human"
    if breathing_root.exists():
        for h5 in breathing_root.rglob("*.h5"):
            path_str = str(h5)
            if "act_empty" in path_str:
                files[0].append(path_str)
            elif "act_sleep" in path_str:
                files[1].append(path_str)

    # ── Source 2: HumanActivityRecognition (all = human present) ──
    har_root = dataset_root / "HumanActivityRecognition" / "sub_Human"
    if har_root.exists():
        # Only use activity folders, skip localization
        activity_prefixes = ["act_jumping", "act_running", "act_seated-breathing",
                             "act_walking-case", "act_wavinghand"]
        for h5 in har_root.rglob("*.h5"):
            path_str = str(h5)
            if any(prefix in path_str for prefix in activity_prefixes):
                files[1].append(path_str)

    return files


def sample_files(files: dict, fraction: float, seed: int) -> tuple:
    """
    Stratified sampling of file paths and labels.

    Returns:
        (file_paths, labels) — lists of sampled paths and integer labels
    """
    rng = random.Random(seed)

    sampled_paths = []
    sampled_labels = []

    for label, paths in files.items():
        n_sample = max(10, int(len(paths) * fraction))
        n_sample = min(n_sample, len(paths))
        selected = rng.sample(paths, n_sample)
        sampled_paths.extend(selected)
        sampled_labels.extend([label] * len(selected))

    return sampled_paths, sampled_labels


# ── Main Training Pipeline ───────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  CSI Human Presence Detection — Training Pipeline")
    print("=" * 65)

    # ── Step 1: Discover dataset ──
    print("\n[1/6] Discovering dataset files...")
    if not DATASET_ROOT.exists():
        print(f"ERROR: Dataset not found at {DATASET_ROOT}")
        sys.exit(1)

    files = discover_files(DATASET_ROOT)
    print(f"  Found {len(files[0]):,} 'no_human' files")
    print(f"  Found {len(files[1]):,} 'human_present' files")
    print(f"  Total: {len(files[0]) + len(files[1]):,} files")

    if len(files[0]) == 0 or len(files[1]) == 0:
        print("ERROR: Need files in both classes. Check dataset path.")
        sys.exit(1)

    # ── Step 2: Sample subset ──
    print(f"\n[2/6] Sampling {SAMPLE_FRACTION*100:.0f}% of files per class...")
    sampled_paths, sampled_labels = sample_files(files, SAMPLE_FRACTION, RANDOM_SEED)
    unique, counts = np.unique(sampled_labels, return_counts=True)
    for u, c in zip(unique, counts):
        label_name = "no_human" if u == 0 else "human_present"
        print(f"  Class {u} ({label_name}): {c:,} files sampled")
    print(f"  Total sampled: {len(sampled_paths):,} files")

    # ── Step 3: Extract features ──
    print(f"\n[3/6] Extracting features (window_size={WINDOW_SIZE})...")
    t0 = time.time()
    X, y = preprocess_batch(sampled_paths, sampled_labels, window_size=WINDOW_SIZE)
    t_extract = time.time() - t0
    print(f"  Feature matrix: {X.shape} ({X.shape[0]:,} windows × {X.shape[1]} features)")
    print(f"  Extraction time: {t_extract:.1f}s")

    # Check for NaN/Inf
    mask = np.isfinite(X).all(axis=1)
    if not mask.all():
        n_bad = (~mask).sum()
        print(f"  Warning: Removing {n_bad} windows with NaN/Inf values")
        X = X[mask]
        y = y[mask]

    # ── Step 4: Normalize ──
    print("\n[4/6] Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {X_train.shape[0]:,} windows")
    print(f"  Test:  {X_test.shape[0]:,} windows")

    # ── Step 5: Train model ──
    print(f"\n[5/6] Training Random Forest (n_estimators={RF_PARAMS['n_estimators']})...")
    t0 = time.time()
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    t_train = time.time() - t0
    print(f"  Training time: {t_train:.1f}s")

    # ── Step 6: Evaluate ──
    print("\n[6/6] Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"\n  Classification Report:")
    target_names = ["no_human", "human_present"]
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  no_human  human_present")
    print(f"  Actual no_human     {cm[0][0]:>5}      {cm[0][1]:>5}")
    print(f"  Actual human        {cm[1][0]:>5}      {cm[1][1]:>5}")

    # Feature importances
    importances = model.feature_importances_
    print(f"\n  Feature Importances:")
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx:
        print(f"    {FEATURE_NAMES[i]:>22s}: {importances[i]:.4f}")

    # ── Save artifacts ──
    print("\n" + "=" * 65)
    print("  Saving model artifacts...")

    model_path = PROJECT_ROOT / "model.pkl"
    scaler_path = PROJECT_ROOT / "scaler.pkl"
    meta_path = PROJECT_ROOT / "model_meta.json"

    joblib.dump(model, model_path)
    print(f"  ✓ Model saved:  {model_path}")

    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Scaler saved: {scaler_path}")

    meta = {
        "task": "csi_human_presence_detection",
        "classes": {0: "no_human", 1: "human_present"},
        "model_type": "RandomForestClassifier",
        "n_features": int(X.shape[1]),
        "feature_names": FEATURE_NAMES,
        "window_size": WINDOW_SIZE,
        "accuracy": round(float(acc), 4),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "sample_fraction": SAMPLE_FRACTION,
        "rf_params": {k: v for k, v in RF_PARAMS.items() if k != "n_jobs"},
        "feature_importances": {
            FEATURE_NAMES[i]: round(float(importances[i]), 4)
            for i in sorted_idx
        },
        "usage": {
            "inference": (
                "from csi_preprocessing import preprocess_realtime_csi; "
                "import joblib; "
                "model = joblib.load('model.pkl'); "
                "scaler = joblib.load('scaler.pkl'); "
                "features = preprocess_realtime_csi(csi_buffer, scaler); "
                "prediction = model.predict(features)"
            ),
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ Metadata saved: {meta_path}")

    print(f"\n  Total pipeline time: {t_extract + t_train:.1f}s")
    print("=" * 65)
    print("  Done! Model ready for deployment.")
    print("=" * 65)


if __name__ == "__main__":
    main()
