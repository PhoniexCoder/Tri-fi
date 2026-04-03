"""
rescue_backend.py — Triangular 3-Node ESP32 Presence Detection Backend

Receives CSI data from 3 ESP32 nodes via UDP (and optionally serial),
runs the TriangularPresenceDetector + CSI-Bench trained ML model,
and broadcasts results to rescue.html over WebSocket on port 8002.

Run from examples/ directory:
    python rescue_backend.py

Nodes send UDP to this machine's IP on port 4444.
rescue.html connects to ws://localhost:8002
"""

import asyncio
import collections
import json
import os
import socket
import sys
import threading
import time
import numpy as np
import websockets
from websockets.server import serve

# Add project root to path so we can import csi_preprocessing
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_inference_triangular import TriangularPresenceDetector
from csi_frame_parser import CSIFrameParser, CSIFrame, VitalsPacket
from csi_preprocessing import preprocess_realtime_csi

# ── Config ────────────────────────────────────────────────────────────────────
UDP_PORT        = 4444
WS_PORT         = 8002
BROADCAST_HZ    = 5       # how often to push to UI (times/sec)
CSI_BUFFER_SIZE = 100     # frames to buffer before ML inference (~3s at 33Hz)
CSI_ML_WEIGHT   = 0.6     # weight of CSI-Bench model in combined probability
TRI_ML_WEIGHT   = 0.4     # weight of triangular detector in combined probability

# ── CSI Amplitude Buffer (per node) ───────────────────────────────────────────

class CSIAmplitudeBuffer:
    """
    Collects CSI amplitude vectors from ESP32 frames and runs the trained
    CSI-Bench Random Forest model when the buffer is full.
    """
    def __init__(self, buffer_size=CSI_BUFFER_SIZE):
        self._buffers = {i: collections.deque(maxlen=buffer_size) for i in [1, 2, 3]}
        self._buffer_size = buffer_size
        self._last_prob = {1: 0.0, 2: 0.0, 3: 0.0}
        self._model = None
        self._scaler = None
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(PROJECT_ROOT, "model.pkl")
        scaler_path = os.path.join(PROJECT_ROOT, "scaler.pkl")
        try:
            import joblib
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self._model = joblib.load(model_path)
                self._scaler = joblib.load(scaler_path)
                print(f"[CSI-ML] Loaded model from {model_path}")
            else:
                print(f"[CSI-ML] WARNING: model.pkl or scaler.pkl not found at project root")
                print(f"[CSI-ML]   Expected: {model_path}")
                print(f"[CSI-ML]   Run train_csi_model.py first.")
        except Exception as e:
            print(f"[CSI-ML] Failed to load model: {e}")

    def push_frame(self, node_id: int, amplitude: np.ndarray):
        """Add a single CSI amplitude vector (n_subcarriers,) to the buffer."""
        if node_id not in self._buffers:
            return
        amp = amplitude.astype(np.float32)
        # Normalize length: if buffer already has frames, match the first frame's length
        buf = self._buffers[node_id]
        if len(buf) > 0:
            target_len = len(buf[0])
            if len(amp) != target_len:
                if len(amp) > target_len:
                    amp = amp[:target_len]       # truncate extra subcarriers
                else:
                    amp = np.pad(amp, (0, target_len - len(amp)))  # zero-pad
        buf.append(amp)

        # Run inference when buffer is full
        if len(buf) >= self._buffer_size:
            self._run_inference(node_id)

    def _run_inference(self, node_id: int):
        """Run the CSI-Bench model on the buffered amplitudes for a node."""
        if self._model is None:
            return

        # Stack frames → (n_subcarriers, n_timesteps)
        buf = list(self._buffers[node_id])
        try:
            amp_matrix = np.stack(buf, axis=1)  # (n_sub, buffer_size)
        except ValueError:
            # Last resort: if shapes still mismatch, clear buffer and skip
            self._buffers[node_id].clear()
            return

        try:
            features = preprocess_realtime_csi(amp_matrix, self._scaler, window_size=self._buffer_size)
            if len(features) > 0:
                proba = self._model.predict_proba(features)
                # Class 1 = human_present probability (average across windows)
                prob = float(np.mean(proba[:, 1]))
                self._last_prob[node_id] = prob
        except Exception as e:
            pass  # Silently skip — don't break the real-time pipeline

        # Clear half the buffer (sliding window, not full reset)
        half = self._buffer_size // 2
        for _ in range(half):
            if self._buffers[node_id]:
                self._buffers[node_id].popleft()

    def get_probability(self, node_id: int) -> float:
        """Get the latest ML probability for a node."""
        return self._last_prob.get(node_id, 0.0)

    def get_combined_probability(self) -> float:
        """Get the max ML probability across all active nodes."""
        return max(self._last_prob.values()) if self._last_prob else 0.0

    @property
    def model_loaded(self) -> bool:
        return self._model is not None


# ── Globals ───────────────────────────────────────────────────────────────────
detector          = TriangularPresenceDetector(meta_path="model_triangular_meta.json")
parser            = CSIFrameParser()
csi_buffer        = CSIAmplitudeBuffer()
connected_clients = set()

# Per-node latest vitals (breathing, motion from VitalsPacket)
_node_vitals: dict = {
    1: {"motion": 0.0, "breathing": 0.0},
    2: {"motion": 0.0, "breathing": 0.0},
    3: {"motion": 0.0, "breathing": 0.0},
}

# ── UDP Receiver (background thread) ─────────────────────────────────────────

def udp_receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(1.0)
    try:
        sock.bind(("0.0.0.0", UDP_PORT))
        print(f"[UDP] Listening on 0.0.0.0:{UDP_PORT}")
    except OSError as e:
        print(f"[UDP] Cannot bind to port {UDP_PORT}: {e}")
        return

    while True:
        try:
            data, _ = sock.recvfrom(4096)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[UDP] error: {e}")
            continue

        result = parser.parse(data)

        if isinstance(result, VitalsPacket):
            nid = result.node_id
            if nid in [1,2,3]:
                print(f"[UDP] Received Vitals Node {nid}: Motion={result.motion_energy:.2f} | Breath={result.breathing_bpm:.1f} BPM")
                _node_vitals[nid]["motion"]    = result.motion_energy
                _node_vitals[nid]["breathing"] = result.breathing_bpm
                detector.update_node(
                    node_id   = nid,
                    rssi      = -70,
                    motion    = result.motion_energy,
                    breathing = result.breathing_bpm,
                )

        elif isinstance(result, CSIFrame):
            nid = result.node_id
            # Log every few frames to avoid spamming
            if detector.n_frames % 50 == 0:
                print(f"[UDP] Receiving CSI from Node {nid} (RSSI: {result.rssi})")
            
            if nid in [1,2,3]:
                # Feed amplitude to CSI-Bench ML buffer
                csi_buffer.push_frame(nid, result.amplitude)

                detector.update_node(
                    node_id   = nid,
                    rssi      = result.rssi,
                    motion    = _node_vitals[nid]["motion"],
                    breathing = _node_vitals[nid]["breathing"],
                )


# ── WebSocket broadcast loop ───────────────────────────────────────────────────

async def broadcast_loop():
    """Compute detection result and push to all connected clients at BROADCAST_HZ."""
    global connected_clients
    interval = 1.0 / BROADCAST_HZ
    while True:
        await asyncio.sleep(interval)
        if not connected_clients:
            continue

        result = detector.compute()

        # ── CSI-Bench ML Model Probability ────────────────────────────────
        csi_ml_prob = csi_buffer.get_combined_probability()
        tri_prob    = result["ml_prob"]

        # Blend: weighted combination of CSI-Bench model + triangular heuristic
        if csi_buffer.model_loaded:
            combined_prob = (CSI_ML_WEIGHT * csi_ml_prob) + (TRI_ML_WEIGHT * tri_prob)
        else:
            combined_prob = tri_prob  # fallback to triangular only

        combined_detected = combined_prob > 0.55

        # Log when detection confidence is meaningful
        if combined_prob > 0.3:
            src = f"CSI-ML:{csi_ml_prob*100:.0f}% + Tri:{tri_prob*100:.0f}%" if csi_buffer.model_loaded else f"Tri:{tri_prob*100:.0f}%"
            print(f"[AI] Combined: {combined_prob*100:.1f}% ({src}) | Detected: {combined_detected}")

        # Per-node display data (now includes CSI-ML per-node probability)
        node_data = []
        for i in [1, 2, 3]:
            node_data.append({
                "node_id":    i,
                "rssi":       result["node_rssi"].get(i, -100),
                "motion":     result["node_motion"].get(i, 0.0),
                "breathing":  result["node_breathing"].get(i, 0.0),
                "active":     i in result["active_node_ids"],
                "csi_ml_prob": round(csi_buffer.get_probability(i), 4),
            })

        payload = json.dumps({
            "type":          "rescue_ai",
            "ai_detected":   combined_detected,
            "ai_prob":       round(combined_prob, 4),
            "ai_mode":       "csi_bench_rf+triangular_v10" if csi_buffer.model_loaded else result["ml_mode"],
            "csi_ml_prob":   round(csi_ml_prob, 4),
            "tri_prob":      round(tri_prob, 4),
            "active_nodes":  result["active_nodes"],
            "pos_x":         result["pos_x"],
            "pos_y":         result["pos_y"],
            "nodes":         node_data,
            # Legacy fields for backwards compat with old rescue.html
            "raw_rssi":      result["node_rssi"].get(1, -100),
            "raw_motion":    result["node_motion"].get(1, 0.0),
            "raw_br":        result["node_breathing"].get(1, 0.0),
            "ai_depth":      0.0,
            "ai_zone":       1 if combined_detected else 0,
        })

        dead = set()
        for client in list(connected_clients):
            try:
                await client.send(payload)
            except Exception:
                dead.add(client)
        connected_clients -= dead


async def ws_handler(websocket):
    connected_clients.add(websocket)
    print(f"[WS] Client connected ({len(connected_clients)} total)")
    try:
        await websocket.wait_closed()
    finally:
        connected_clients.discard(websocket)
        print(f"[WS] Client disconnected ({len(connected_clients)} total)")


async def main():
    print("=" * 55)
    print("  Rescue Backend — Hybrid Presence Detection System")
    print("=" * 55)
    print(f"  UDP port:  {UDP_PORT}  (ESP32 nodes → this machine)")
    print(f"  WS  port:  {WS_PORT}   (rescue.html ← this machine)")
    print(f"  CSI Model: {'LOADED ✓' if csi_buffer.model_loaded else 'NOT FOUND ✗ (run train_csi_model.py)'}")
    print(f"  Blend:     CSI-ML {CSI_ML_WEIGHT*100:.0f}% + Triangular {TRI_ML_WEIGHT*100:.0f}%")
    print("=" * 55)

    # Start UDP receiver thread
    t = threading.Thread(target=udp_receiver, daemon=True)
    t.start()

    # Start WebSocket server + broadcast loop
    async with serve(ws_handler, "0.0.0.0", WS_PORT):
        print(f"[WS] Server ready at ws://localhost:{WS_PORT}")
        print(f"     Open rescue.html in your browser now.\n")
        await broadcast_loop()


if __name__ == "__main__":
    asyncio.run(main())
