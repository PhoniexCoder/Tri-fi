#!/usr/bin/env python3
"""
capture_dataset.py — Real CSI Data Capture for Model Retraining

Listens for live UDP frames from your ESP32 nodes, extracts the same 27
features used during training, and saves them to a labeled CSV.

WORKFLOW:
  1. Make sure your ESP32s are running and sending UDP to this machine.
  2. Run this script and follow the on-screen prompts.
  3. For each session, enter the label (zone) and stand/sit in that zone.
  4. Press ENTER to stop that label session, then move to the next zone.
  5. When done, a dataset CSV is saved — retrain with ml_train_rf.py.

Usage:
    python capture_dataset.py --udp-port 4444 --output real_dataset.csv

Requirements:
    pip install numpy  (already installed)
"""

import argparse
import collections
import csv
import os
import socket
import sys
import threading
import time
from datetime import datetime

import numpy as np

# ── Import project modules (must run from examples/ directory) ─────────────
try:
    from csi_frame_parser import CSIFrameParser, CSIFrame, VitalsPacket
    from csi_feature_extractor import CSIFeatureExtractor
except ImportError:
    print("ERROR: Run this script from the examples/ directory:")
    print("  cd examples")
    print("  python capture_dataset.py")
    sys.exit(1)

# ── Feature column names (must match ml_train_rf.py exactly) ──────────────
FEATURE_COLS = (
    ["rssi", "motion", "breathing"] +
    [f"amp_pca_{i}"   for i in range(10)] +
    [f"phase_pca_{i}" for i in range(10)] +
    ["amp_var_mean", "amp_var_std", "peak_sub_norm", "snr_db"]
)
ALL_COLS = FEATURE_COLS + ["label", "breathing_depth", "body_zone", "timestamp", "node_id"]

ZONE_NAMES = {
    0: "ABSENT   (no person — leave the room)",
    1: "NEAR     (< 1 m from ESP32)",
    2: "MID      (1–3 m from ESP32)",
    3: "FAR      (> 3 m from ESP32)",
}

# ─────────────────────────────────────────────────────────────────────────────

class DataCapture:
    def __init__(self, udp_port: int, output_csv: str):
        self.udp_port   = udp_port
        self.output_csv = output_csv

        self._parser    = CSIFrameParser()
        self._extractor = CSIFeatureExtractor()

        # Latest vitals from VitalsPacket (updated by UDP receiver)
        self._latest_motion    = 0.0
        self._latest_breathing = 0.0
        self._latest_node_id   = 0

        # Capture state
        self._capturing    = False
        self._current_zone = 0
        self._rows_buffer  = []
        self._lock         = threading.Lock()
        self._stop_event   = threading.Event()

        # Stats
        self.total_frames  = 0
        self.total_rows    = 0

    # ──────────────────────────────────────────────────────────────────────────
    # UDP receiver (runs in background thread)
    # ──────────────────────────────────────────────────────────────────────────

    def _udp_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.5)
        try:
            sock.bind(("0.0.0.0", self.udp_port))
        except OSError as e:
            print(f"\n[UDP] Cannot bind to port {self.udp_port}: {e}")
            print(f"      Is another process using this port? Try --udp-port 4445")
            self._stop_event.set()
            return

        print(f"[UDP] Listening on 0.0.0.0:{self.udp_port} ...")

        while not self._stop_event.is_set():
            try:
                data, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"[UDP] recv error: {e}")
                continue

            result = self._parser.parse(data)

            if isinstance(result, VitalsPacket):
                with self._lock:
                    self._latest_motion    = result.motion_energy
                    self._latest_breathing = result.breathing_bpm
                    self._latest_node_id   = result.node_id

            elif isinstance(result, CSIFrame):
                self.total_frames += 1
                with self._lock:
                    motion    = self._latest_motion
                    breathing = self._latest_breathing
                    node_id   = result.node_id or self._latest_node_id

                # Feature extraction (lock-free — extractor has its own state)
                vec = self._extractor.update(result, result.rssi, motion, breathing)

                if vec is None:
                    # PCA still warming up
                    continue

                if self._capturing:
                    zone  = self._current_zone
                    label = 1 if zone > 0 else 0
                    # breathing_depth: use normalised breathing as proxy
                    # (real depth estimation needs the RF model — here we
                    #  use a simple heuristic: higher breathing = deeper)
                    br_norm = float(np.clip(breathing / 30.0, 0.0, 1.0))
                    depth_proxy = float(np.clip(br_norm * (1.0 - abs(result.rssi + 70) / 50.0), 0.0, 1.0))

                    row = list(vec) + [label, depth_proxy, zone,
                                       time.time(), node_id]
                    with self._lock:
                        self._rows_buffer.append(row)
                    self.total_rows += 1

        sock.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Interactive capture session
    # ──────────────────────────────────────────────────────────────────────────

    def run(self):
        # Start UDP thread
        udp_thread = threading.Thread(target=self._udp_loop, daemon=True)
        udp_thread.start()

        # Wait for UDP to bind
        time.sleep(1.0)
        if self._stop_event.is_set():
            return

        print("\n" + "="*60)
        print("  ESP32 CSI Data Capture Tool")
        print("="*60)
        print(f"  Output: {self.output_csv}")
        print(f"  UDP Port: {self.udp_port}")
        print("="*60)
        print("\nWaiting for CSI frames... (PCA needs ~200 frames to warm up)\n")

        # Wait for PCA warm-up
        warmup_start = time.time()
        while not self._extractor.is_ready:
            elapsed = time.time() - warmup_start
            frames  = self._parser.n_parsed
            print(f"\r  Warming up: {frames} frames received ({elapsed:.0f}s)...", end="", flush=True)
            if self._stop_event.is_set():
                return
            time.sleep(0.5)
            if elapsed > 60:
                print("\n\n[WARN] No UDP frames received after 60s!")
                print("  → Check ESP32 WiFi / target IP / UDP port settings")
                print("  → Is your ESP32 sending to this machine's IP?")
                self._run_without_pca()
                return

        print(f"\n\n✓ PCA ready! ({self._parser.n_parsed} frames received)\n")

        all_rows = []

        for zone in [0, 1, 2, 3]:
            print("─"*60)
            print(f"  ZONE {zone}: {ZONE_NAMES[zone]}")
            print("─"*60)
            input(f"  Position yourself, then press ENTER to start recording...")

            # Start capture
            with self._lock:
                self._rows_buffer.clear()
            self._current_zone = zone
            self._capturing    = True

            # Live status display
            start_t = time.time()
            try:
                while True:
                    elapsed = time.time() - start_t
                    with self._lock:
                        n = len(self._rows_buffer)
                    print(f"\r  Recording zone {zone}: {n} frames ({elapsed:.0f}s) — "
                          f"Press ENTER to stop", end="", flush=True)
                    time.sleep(0.2)
            except KeyboardInterrupt:
                pass

            # Use a thread to wait for ENTER while showing live counter
            stop_flag = threading.Event()

            def _wait_enter():
                input()
                stop_flag.set()

            enter_thread = threading.Thread(target=_wait_enter, daemon=True)
            enter_thread.start()

            while not stop_flag.is_set():
                elapsed = time.time() - start_t
                with self._lock:
                    n = len(self._rows_buffer)
                print(f"\r  Recording zone {zone}: {n} frames ({elapsed:.0f}s) — "
                      f"Press ENTER to stop", end="", flush=True)
                time.sleep(0.2)

            self._capturing = False
            with self._lock:
                captured = list(self._rows_buffer)
                self._rows_buffer.clear()

            print(f"\n  ✓ Captured {len(captured)} frames for zone {zone}")
            if len(captured) < 100:
                print(f"  [WARN] Only {len(captured)} frames — recommend 200+ per zone for good accuracy")

            all_rows.extend(captured)

            if zone < 3:
                cont = input(f"\n  Continue to next zone? [Y/n]: ").strip().lower()
                if cont == "n":
                    break

        # Stop UDP
        self._stop_event.set()

        # Save CSV
        self._save(all_rows)

    def _save(self, all_rows: list):
        if not all_rows:
            print("\n[WARN] No data captured — nothing saved.")
            return

        # Merge with existing CSV if it exists
        existing_rows = []
        if os.path.exists(self.output_csv):
            merge = input(f"\n  '{self.output_csv}' already exists. Append to it? [Y/n]: ").strip().lower()
            if merge != "n":
                import csv as csv_mod
                with open(self.output_csv, "r", newline="") as f:
                    reader = csv_mod.DictReader(f)
                    for row in reader:
                        existing_rows.append([row.get(c, 0) for c in ALL_COLS])
                print(f"  Loaded {len(existing_rows)} existing rows.")

        combined = existing_rows + all_rows
        with open(self.output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(ALL_COLS)
            writer.writerows(combined)

        # Summary
        import numpy as np
        data = np.array([r[:len(FEATURE_COLS)+3] for r in combined])
        zones = data[:, -1].astype(int)
        print(f"\n{'='*60}")
        print(f"  Dataset saved: {self.output_csv}")
        print(f"  Total rows: {len(combined):,}")
        print(f"  Frames per zone:")
        for z in range(4):
            n = int((zones == z).sum())
            print(f"    Zone {z} ({ZONE_NAMES[z][:8]}): {n:,} frames")
        print(f"\n  Next step — retrain the model:")
        print(f"    python ml_train_rf.py --data {self.output_csv} --prefix model_rf")
        print(f"{'='*60}\n")

    def _run_without_pca(self):
        """Fallback guidance if no UDP frames arrive."""
        print("\n  No live UDP data detected. Possible fixes:")
        print("")
        print("  1. Check ESP32 target IP is set to THIS machine's IP:")
        print("     python provision.py --port COM<X> --target-ip <YOUR_IP> --target-port 4444")
        print("")
        print("  2. Check Windows Firewall allows UDP on port 4444:")
        print("     netsh advfirewall firewall add rule name='ESP32 CSI' dir=in")
        print("     action=allow protocol=UDP localport=4444")
        print("")
        print("  3. Verify ESP32 is connected to same WiFi network as laptop.")
        print("")
        print("  4. Run trifi_live.py first and check if CSI data appears in the UI.")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Capture real CSI data from ESP32 nodes for RF model retraining",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python capture_dataset.py                          # default port 4444
  python capture_dataset.py --udp-port 5005          # custom port
  python capture_dataset.py --output my_room.csv     # custom output file
        """,
    )
    ap.add_argument("--udp-port", type=int, default=4444,
                    help="UDP port to listen on (must match ESP32 target_port, default: 4444)")
    ap.add_argument("--output",   default="real_dataset.csv",
                    help="Output CSV file (default: real_dataset.csv)")
    args = ap.parse_args()

    cap = DataCapture(udp_port=args.udp_port, output_csv=args.output)
    try:
        cap.run()
    except KeyboardInterrupt:
        print("\n\nCapture interrupted by user.")
        cap._stop_event.set()


if __name__ == "__main__":
    main()
