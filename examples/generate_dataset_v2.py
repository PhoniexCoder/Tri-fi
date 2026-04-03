#!/usr/bin/env python3
"""
generate_dataset_v2.py — Synthetic Dataset Generator (v2)

Generates a CSV with all 27 features + 3 target columns:
  label, breathing_depth, body_zone

Body zones (distance from node):
  0 = absent   (no person)
  1 = near     (<1 m)
  2 = mid      (1–3 m)
  3 = far      (>3 m)

Physics simulation:
  - Near:  high-variance amplitude, strong phase oscillation, high RSSI
  - Mid:   moderate amplitude variance, moderate phase, medium RSSI
  - Far:   low variance, small phase shift, weak RSSI
  - Absent: near-zero motion, no breathing, weak RSSI with no periodicity

Usage:
    python generate_dataset_v2.py --rows 8000 --output dataset_v2.csv
"""

import argparse
import numpy as np
import pandas as pd

N_AMP   = 10
N_PHASE = 10


def simulate_zone(rng, zone: int, n: int):
    """Generate realistic feature distributions for each body zone."""

    if zone == 0:   # ── Absent ─────────────────────────────────────────────
        rssi      = rng.normal(-84, 4,  n).clip(-100, -65)
        motion    = rng.beta(1, 15, n).clip(0, 0.04)
        breathing = rng.normal(0, 0.4,  n).clip(0, 1.5)
        depth     = np.zeros(n)
        amp_pca   = rng.normal(0,    0.2, (n, N_AMP))
        phase_pca = rng.normal(0,    0.15,(n, N_PHASE))
        amp_var_mean = rng.beta(1, 10, n) * 0.1
        amp_var_std  = rng.beta(1, 12, n) * 0.05
        peak_sub     = rng.uniform(0,    1,   n)   # random (no body)
        snr_db       = rng.normal(0.15, 0.05, n).clip(0.05, 0.3)

    elif zone == 1: # ── Near (<1 m) ─────────────────────────────────────────
        rssi      = rng.normal(-56, 4,  n).clip(-80, -35)
        motion    = rng.beta(3,  5,  n).clip(0.08, 0.6)
        breathing = rng.normal(16, 3,   n).clip(8,  28)
        depth     = rng.beta(4, 2, n).clip(0.5, 1.0)         # deep breaths nearby
        amp_pca   = rng.normal([1.2]+[0]*(N_AMP-1), 0.3, (n, N_AMP))
        phase_pca = rng.normal([-0.8]+[0]*(N_PHASE-1), 0.25, (n, N_PHASE))
        amp_var_mean = rng.beta(4, 3, n) * 0.7 + 0.3
        amp_var_std  = rng.beta(3, 4, n) * 0.4 + 0.1
        peak_sub     = rng.normal(0.25, 0.1, n).clip(0.05, 0.5)  # low-index subs
        snr_db       = rng.normal(0.75, 0.08, n).clip(0.5, 1.0)

    elif zone == 2: # ── Mid (1–3 m) ──────────────────────────────────────────
        rssi      = rng.normal(-68, 5,  n).clip(-85, -50)
        motion    = rng.beta(2,  6,  n).clip(0.03, 0.3)
        breathing = rng.normal(14, 3,   n).clip(6,  26)
        depth     = rng.beta(3, 3, n).clip(0.2, 0.8)
        amp_pca   = rng.normal([0.5]+[0]*(N_AMP-1), 0.25, (n, N_AMP))
        phase_pca = rng.normal([-0.3]+[0]*(N_PHASE-1), 0.2, (n, N_PHASE))
        amp_var_mean = rng.beta(3, 4, n) * 0.4 + 0.1
        amp_var_std  = rng.beta(2, 5, n) * 0.2 + 0.05
        peak_sub     = rng.normal(0.5, 0.15, n).clip(0.2, 0.8)   # mid-index subs
        snr_db       = rng.normal(0.5, 0.1, n).clip(0.25, 0.75)

    else:           # ── Far (>3 m) ───────────────────────────────────────────
        rssi      = rng.normal(-78, 4,  n).clip(-95, -60)
        motion    = rng.beta(1,  8,  n).clip(0.01, 0.12)
        breathing = rng.normal(12, 4,   n).clip(4,  20)
        depth     = rng.beta(2, 5, n).clip(0.05, 0.4)
        amp_pca   = rng.normal([0.2]+[0]*(N_AMP-1), 0.15, (n, N_AMP))
        phase_pca = rng.normal([-0.1]+[0]*(N_PHASE-1), 0.12, (n, N_PHASE))
        amp_var_mean = rng.beta(2, 6, n) * 0.2 + 0.02
        amp_var_std  = rng.beta(1, 8, n) * 0.1 + 0.01
        peak_sub     = rng.normal(0.7, 0.15, n).clip(0.4, 0.95)  # high-index subs
        snr_db       = rng.normal(0.3, 0.08, n).clip(0.1, 0.55)

    rows = {
        "rssi":      rssi,
        "motion":    motion,
        "breathing": breathing,
        "label":     np.ones(n, dtype=int) if zone > 0 else np.zeros(n, dtype=int),
        "breathing_depth": depth,
        "body_zone": np.full(n, zone, dtype=int),
    }
    for i in range(N_AMP):
        rows[f"amp_pca_{i}"]   = amp_pca[:, i]
    for i in range(N_PHASE):
        rows[f"phase_pca_{i}"] = phase_pca[:, i]
    rows["amp_var_mean"] = amp_var_mean
    rows["amp_var_std"]  = amp_var_std
    rows["peak_sub_norm"]= peak_sub
    rows["snr_db"]       = snr_db

    return rows


def generate(n_rows: int, output: str, seed: int = 42):
    rng = np.random.default_rng(seed)

    n_per_zone = n_rows // 4
    remainder  = n_rows - n_per_zone * 4

    all_rows = []
    for zone in range(4):
        n = n_per_zone + (1 if zone < remainder else 0)
        rows = simulate_zone(rng, zone, n)
        all_rows.append(pd.DataFrame(rows))

    df = pd.concat(all_rows, ignore_index=True)

    # ── Inject ~2% NaN (realistic sensor dropout) ────────────────────────────
    feature_cols = (
        ["rssi", "motion", "breathing"] +
        [f"amp_pca_{i}" for i in range(N_AMP)] +
        [f"phase_pca_{i}" for i in range(N_PHASE)] +
        ["amp_var_mean", "amp_var_std", "peak_sub_norm", "snr_db"]
    )
    for col in feature_cols:
        mask = rng.random(len(df)) < 0.02
        df.loc[mask, col] = np.nan

    # ── Shuffle ───────────────────────────────────────────────────────────────
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    df.to_csv(output, index=False)
    print(f"[dataset_v2] {len(df):,} rows → {output}")
    print(f"  Zone distribution: {df['body_zone'].value_counts().sort_index().to_dict()}")
    print(f"  Presence:  {df['label'].sum():,} / {len(df):,}")
    print(f"  Breath depth (mean by zone):")
    for z in range(4):
        sub = df[df["body_zone"] == z]
        print(f"    zone {z}: {sub['breathing_depth'].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows",   type=int, default=8000)
    ap.add_argument("--output", default="dataset_v2.csv")
    args = ap.parse_args()
    generate(args.rows, args.output)
