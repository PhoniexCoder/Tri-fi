#!/usr/bin/env python3
"""
ml_train_v2.py — Upgraded Multi-Output LSTM Training Pipeline
Uses 27 raw-CSI features (amplitude PCA + phase PCA + summary + vitals).

Outputs (3 simultaneous prediction heads):
  1. presence_prob   (0–1) — binary: human present / absent
  2. breathing_depth (0–1) — relative breathing depth (tidal volume proxy)
  3. body_zone       (0–3, one-hot) — coarse position: near/mid/far/out-of-range

Input shape: (SEQ_LEN=20, N_FEATURES=27)

CSV format (extended from v1):
  rssi, motion, breathing, label,
  amp_pca_0..9,          ← 10 amplitude PCA columns
  phase_pca_0..9,        ← 10 phase-diff PCA columns
  amp_var_mean, amp_var_std, peak_sub_norm, snr_db,  ← 4 summary stats
  breathing_depth,       ← 0.0–1.0 (0 if empty room)
  body_zone              ← 0=absent, 1=near(<1m), 2=mid(1-3m), 3=far(>3m)

Usage:
    python ml_train_v2.py --data dataset_v2.csv --output model_v2.h5
"""

import argparse
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, Lambda
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN    = 20
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

BATCH_SIZE = 32
EPOCHS     = 120
LR         = 8e-4
N_ZONES    = 4   # 0=absent, 1=near, 2=mid, 3=far


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = FEATURE_COLS + ["label", "breathing_depth", "body_zone"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    df[FEATURE_COLS]  = df[FEATURE_COLS].ffill().fillna(0.0)
    df["label"]           = df["label"].fillna(0).astype(int)
    df["breathing_depth"] = df["breathing_depth"].fillna(0.0).clip(0.0, 1.0)
    df["body_zone"]       = df["body_zone"].fillna(0).astype(int).clip(0, N_ZONES - 1)

    df["rssi"]    = df["rssi"].clip(-100, -20)
    df["motion"]  = df["motion"].clip(0.0, 1.0)

    print(f"[data] {len(df):,} rows | "
          f"present={df['label'].sum():,} | "
          f"zones={df['body_zone'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. SEQUENCE CREATION
# ─────────────────────────────────────────────────────────────────────────────

def make_sequences(X, y_pres, y_depth, y_zone):
    Xs, Yp, Yd, Yz = [], [], [], []
    for i in range(len(X) - SEQ_LEN + 1):
        Xs.append(X[i : i + SEQ_LEN])
        Yp.append(y_pres[i  + SEQ_LEN - 1])
        Yd.append(y_depth[i + SEQ_LEN - 1])
        Yz.append(y_zone[i  + SEQ_LEN - 1])
    return (np.array(Xs, np.float32),
            np.array(Yp, np.float32),
            np.array(Yd, np.float32),
            tf.keras.utils.to_categorical(Yz, N_ZONES).astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# 3. MODEL ARCHITECTURE — multi-output LSTM
# ─────────────────────────────────────────────────────────────────────────────

def build_model_v2(seq_len: int, n_features: int) -> Model:
    """
    Shared LSTM trunk → 3 independent prediction heads.

    Trunk:
      LSTM(128) → BN → Dropout(0.3) → LSTM(64) → Dropout(0.2) → Dense(32,relu)

    Heads:
      presence_head   : Dense(1, sigmoid)   — binary classification
      depth_head      : Dense(1, sigmoid)   — regression [0,1]
      zone_head       : Dense(4, softmax)   — 4-class zone classification

    Why a shared trunk?
      Breathing depth and body position are correlated with presence.
      Shared weights learn a common "is there a person + where + how they breathe"
      representation, which regularises all three tasks simultaneously.
    """
    inp = Input(shape=(seq_len, n_features), name="csi_input")

    # ── Shared LSTM trunk ─────────────────────────────────────────────────────
    x = LSTM(128, return_sequences=True, name="lstm_1")(inp)
    x = BatchNormalization(name="bn_1")(x)
    x = Dropout(0.30, name="drop_1")(x)
    x = LSTM(64,  return_sequences=False, name="lstm_2")(x)
    x = Dropout(0.20, name="drop_2")(x)
    shared = Dense(32, activation="relu", name="shared_fc")(x)

    # ── Head 1: presence probability ─────────────────────────────────────────
    presence = Dense(16, activation="relu",    name="p_fc")(shared)
    presence = Dense(1,  activation="sigmoid", name="presence")(presence)

    # ── Head 2: breathing depth [0,1] ─────────────────────────────────────────
    depth = Dense(16, activation="relu",    name="d_fc")(shared)
    depth = Dense(1,  activation="sigmoid", name="breathing_depth")(depth)

    # ── Head 3: body zone (4-class) ───────────────────────────────────────────
    zone = Dense(16, activation="relu",    name="z_fc")(shared)
    zone = Dense(N_ZONES, activation="softmax", name="body_zone")(zone)

    model = Model(inputs=inp, outputs=[presence, depth, zone],
                  name="survivor_v2")

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss={
            "presence":        "binary_crossentropy",
            "breathing_depth": "mse",
            "body_zone":       "categorical_crossentropy",
        },
        loss_weights={
            "presence":        1.0,   # primary objective
            "breathing_depth": 0.5,   # auxiliary
            "body_zone":       0.5,   # auxiliary
        },
        metrics={
            "presence":        [tf.keras.metrics.AUC(name="auc"),
                                tf.keras.metrics.BinaryAccuracy(name="acc")],
            "breathing_depth": ["mae"],
            "body_zone":       ["accuracy"],
        },
    )
    model.summary()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLASS-WEIGHT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def class_weights(y):
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    n_tot = len(y)
    return {0: n_tot / (2 * n_neg + 1e-9), 1: n_tot / (2 * n_pos + 1e-9)}


# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train(csv_path, model_path, scaler_path, meta_path):
    df = load_and_clean(csv_path)

    train_df, test_df = train_test_split(df, test_size=0.2,
                                         shuffle=True, random_state=42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train_df[FEATURE_COLS].values).astype(np.float32)
    X_te = scaler.transform(test_df[FEATURE_COLS].values).astype(np.float32)
    joblib.dump(scaler, scaler_path)
    print(f"[scaler] → {scaler_path}")

    Xtr, Yp_tr, Yd_tr, Yz_tr = make_sequences(
        X_tr,
        train_df["label"].values,
        train_df["breathing_depth"].values,
        train_df["body_zone"].values,
    )
    Xte, Yp_te, Yd_te, Yz_te = make_sequences(
        X_te,
        test_df["label"].values,
        test_df["breathing_depth"].values,
        test_df["body_zone"].values,
    )
    print(f"[seqs] train={Xtr.shape} test={Xte.shape}")

    model = build_model_v2(SEQ_LEN, N_FEATURES)
    cw    = class_weights(Yp_tr)

    callbacks = [
        EarlyStopping(monitor="val_presence_auc", patience=15,
                      mode="max", restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor="val_presence_auc",
                        save_best_only=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=7, min_lr=1e-6, verbose=1),
    ]

    model.fit(
        Xtr,
        {"presence": Yp_tr, "breathing_depth": Yd_tr, "body_zone": Yz_tr},
        validation_data=(
            Xte,
            {"presence": Yp_te, "breathing_depth": Yd_te, "body_zone": Yz_te},
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    results = model.evaluate(
        Xte,
        {"presence": Yp_te, "breathing_depth": Yd_te, "body_zone": Yz_te},
        verbose=0,
    )
    print("\n[eval]", dict(zip(model.metrics_names, results)))

    meta = {
        "version":      2,
        "seq_len":      SEQ_LEN,
        "n_features":   N_FEATURES,
        "feature_cols": FEATURE_COLS,
        "n_zones":      N_ZONES,
        "model_path":   model_path,
        "scaler_path":  scaler_path,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[meta] → {meta_path}")
    print(f"\n✓ Model v2 saved → {model_path}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default="dataset_v2.csv")
    ap.add_argument("--output", default="model_v2.h5")
    ap.add_argument("--scaler", default="scaler_v2.pkl")
    ap.add_argument("--meta",   default="model_v2_meta.json")
    args = ap.parse_args()
    train(args.data, args.output, args.scaler, args.meta)
