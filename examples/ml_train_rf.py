#!/usr/bin/env python3
"""
ml_train_rf.py — Traditional ML Fallback Pipeline (Random Forest)
Trains independent RF models for presence, breathing depth, and body zone.
"""

import argparse
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error, classification_report

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
N_AMP      = 10
N_PHASE    = 10
N_SUMMARY  = 4
N_ORIGINAL = 3
N_FEATURES = N_ORIGINAL + N_AMP + N_PHASE + N_SUMMARY   # 27

FEATURE_COLS = (
    ["rssi", "motion", "breathing"] +
    [f"amp_pca_{i}"   for i in range(N_AMP)]   +
    [f"phase_pca_{i}" for i in range(N_PHASE)] +
    ["amp_var_mean", "amp_var_std", "peak_sub_norm", "snr_db"]
)

N_ZONES = 4

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = FEATURE_COLS + ["label", "breathing_depth", "body_zone"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    # Drop extra columns that capture_dataset.py adds (timestamp, node_id)
    extra = [c for c in df.columns if c not in FEATURE_COLS + ["label", "breathing_depth", "body_zone"]]
    if extra:
        print(f"[data] Dropping extra columns: {extra}")
        df = df.drop(columns=extra)

    df[FEATURE_COLS]      = df[FEATURE_COLS].ffill().fillna(0.0)
    df["label"]           = df["label"].fillna(0).astype(int)
    df["breathing_depth"] = df["breathing_depth"].fillna(0.0).clip(0.0, 1.0)
    df["body_zone"]       = df["body_zone"].fillna(0).astype(int).clip(0, N_ZONES - 1)

    # NOTE: rssi column is already normalised to [0,1] by CSIFeatureExtractor.
    # (map: [-100,-30] -> [0,1] via (rssi+100)/70). Do NOT clip to raw dBm.
    df["rssi"]    = df["rssi"].clip(0.0, 1.0)
    df["motion"]  = df["motion"].clip(0.0, 1.0)
    df["breathing"] = df["breathing"].clip(0.0, 1.0)

    n_present = df['label'].sum()
    n_absent  = len(df) - n_present
    print(f"[data] {len(df):,} rows | present={n_present:,} | absent={n_absent:,}")
    print(f"[data] zones={df['body_zone'].value_counts().sort_index().to_dict()}")
    if n_present == 0 or n_absent == 0:
        raise ValueError("Dataset must have both present and absent examples!")
    return df

def train(csv_path, prefix_out):
    df = load_and_clean(csv_path)

    X = df[FEATURE_COLS].values
    y_pres  = df["label"].values
    y_depth = df["breathing_depth"].values
    y_zone  = df["body_zone"].values

    # Train / Test split
    X_train, X_test, y_pres_tr, y_pres_te, y_depth_tr, y_depth_te, y_zone_tr, y_zone_te = train_test_split(
        X, y_pres, y_depth, y_zone, test_size=0.2, shuffle=True, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)
    
    scaler_path = f"{prefix_out}_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"[scaler] Saved to {scaler_path}")

    # 1. Presence (Binary Classifier)
    print("\n--- Training Presence Model ---")
    # Class weights balance present/absent even if counts are unequal
    clf_pres = RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced",
        random_state=42, n_jobs=-1)
    clf_pres.fit(X_train_s, y_pres_tr)
    y_pres_pred = clf_pres.predict(X_test_s)
    y_pres_prob = clf_pres.predict_proba(X_test_s)[:, 1]

    print(f"Presence Accuracy: {accuracy_score(y_pres_te, y_pres_pred):.4f}")
    if len(np.unique(y_pres_te)) > 1:
        print(f"Presence AUC:      {roc_auc_score(y_pres_te, y_pres_prob):.4f}")
    print(classification_report(y_pres_te, y_pres_pred,
          target_names=["absent", "present"], zero_division=0))
    model_pres_path = f"{prefix_out}_presence.pkl"
    joblib.dump(clf_pres, model_pres_path)
    print(f"[saved] {model_pres_path}")

    # Top 5 most important features
    feat_imp = sorted(zip(FEATURE_COLS, clf_pres.feature_importances_),
                      key=lambda x: x[1], reverse=True)[:5]
    print("  Top features:", [(f, f"{v:.3f}") for f, v in feat_imp])

    # 2. Breathing Depth (Regressor)
    print("\n--- Training Breathing Depth Model ---")
    reg_depth = RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    reg_depth.fit(X_train_s, y_depth_tr)
    y_depth_pred = reg_depth.predict(X_test_s)

    print(f"Depth MAE: {mean_absolute_error(y_depth_te, y_depth_pred):.4f}")
    model_depth_path = f"{prefix_out}_depth.pkl"
    joblib.dump(reg_depth, model_depth_path)
    print(f"[saved] {model_depth_path}")

    # 3. Body Zone (Multiclass Classifier)
    print("\n--- Training Body Zone Model ---")
    clf_zone = RandomForestClassifier(
        n_estimators=200, max_depth=15, class_weight="balanced",
        random_state=42, n_jobs=-1)
    clf_zone.fit(X_train_s, y_zone_tr)
    y_zone_pred = clf_zone.predict(X_test_s)

    print(f"Zone Accuracy: {accuracy_score(y_zone_te, y_zone_pred):.4f}")
    print("Zone Classification Report:")
    print(classification_report(y_zone_te, y_zone_pred,
          target_names=["absent","near","mid","far"], zero_division=0))
    model_zone_path = f"{prefix_out}_zone.pkl"
    joblib.dump(clf_zone, model_zone_path)
    print(f"[saved] {model_zone_path}")

    # Meta
    meta = {
        "version":      "rf_1",
        "n_features":   N_FEATURES,
        "feature_cols": FEATURE_COLS,
        "n_zones":      N_ZONES,
        "scaler_path":  scaler_path,
        "model_presence": model_pres_path,
        "model_depth":  model_depth_path,
        "model_zone":   model_zone_path,
    }
    meta_path = f"{prefix_out}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[meta] Done. Metadata saved to {meta_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default="dataset_v2.csv")
    ap.add_argument("--prefix", default="model_rf")
    args = ap.parse_args()
    train(args.data, args.prefix)
