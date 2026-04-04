# visualize_custom_dataset.py
import os
import glob
import re
import random

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3D projection

DATA_DIR = "custom_dataset"
WINDOW_SIZE = 100

label_re = re.compile(r"_label(\d+)_")

def parse_label(path: str) -> int | None:
    m = label_re.search(os.path.basename(path))
    return int(m.group(1)) if m else None

def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.npy"))
    if not files:
        print("No .npy files found in custom_dataset/")
        return

    file_counts = {}
    window_counts = {}
    examples = {}

    for f in files:
        label = parse_label(f)
        if label is None:
            continue
        frames = np.load(f)  # shape: (time, 64)
        T = frames.shape[0]

        file_counts[label] = file_counts.get(label, 0) + 1

        # approximate number of 100-frame windows with 50% overlap (like training)
        step = WINDOW_SIZE // 2
        n_windows = max(0, (T - WINDOW_SIZE) // step + 1)
        window_counts[label] = window_counts.get(label, 0) + n_windows

        # keep one example per label to visualize
        if label not in examples and T >= WINDOW_SIZE:
            start = 0 if T == WINDOW_SIZE else random.randint(0, T - WINDOW_SIZE)
            window = frames[start:start + WINDOW_SIZE].T  # (64, 100)
            examples[label] = window

    labels_sorted = sorted(examples.keys())
    if not labels_sorted:
        print("No windows with at least", WINDOW_SIZE, "frames to visualise")
        return

    # ── 3D Constellation-style scatter plots ──────────────────────────────
    # We only have amplitudes, not true complex CSI. To get a similar
    # look-and-feel to I/Q constellations, we take pairs of consecutive
    # samples from one window and plot (even, odd) as (x, y), with
    # sample index along the z-axis.

    n_plots = min(4, len(labels_sorted))
    plt.figure(figsize=(10, 8))

    for idx in range(n_plots):
        label = labels_sorted[idx]
        win = examples[label]  # (64, 100)
        flat = win.reshape(-1)
        if flat.size < 2:
            continue

        real = flat[0::2]
        imag = flat[1::2]
        m = min(len(real), len(imag))
        real = real[:m]
        imag = imag[:m]

        # z-axis: sample index (0..m-1)
        idxs = np.arange(m)

        ax = plt.subplot(2, 2, idx + 1, projection="3d")
        ax.scatter(real, imag, idxs, s=6, alpha=0.5, depthshade=True)
        ax.set_xlabel("Real")
        ax.set_ylabel("Imag")
        ax.set_zlabel("Sample Index")
        ax.set_title(f"Label {label}")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()