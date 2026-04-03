"""
rescue_backend.py — Complete CSI Human Detection Backend
=========================================================
3 ESP32-S3 nodes (triangular) → UDP → Feature Extraction + Breathing FFT
→ Random Forest → Triangle Consensus → WebSocket → rescue.html heatmap

Run:  python rescue_backend.py
Nodes send UDP to port 4444. UI connects to ws://localhost:8002
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

# Add project root for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_inference_triangular import TriangularPresenceDetector
from csi_frame_parser import CSIFrameParser, CSIFrame, VitalsPacket
from csi_preprocessing import preprocess_realtime_csi, get_breathing_info

# Suppress deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import websockets
from websockets.server import serve


# ── Config ────────────────────────────────────────────────────────────────────

UDP_PORT        = 4444
WS_PORT         = 8002
BROADCAST_HZ    = 5
CSI_BUFFER_SIZE = 100     # frames before ML inference (~3s at 33Hz)
CSI_ML_WEIGHT   = 0.6     # CSI-Bench model weight in blend
TRI_ML_WEIGHT   = 0.4     # Triangular heuristic weight
MIN_CONSENSUS   = 2       # minimum nodes detecting for triangle consensus

# Triangle node positions (fixed layout)
NODE_POSITIONS = {
    1: (0.0, 0.0),
    2: (1.0, 0.0),
    3: (0.5, 1.0),
}


# ── CSI Amplitude Buffer + ML Inference ──────────────────────────────────────

class CSINodeInference:
    """
    Per-node CSI amplitude buffer with ML inference and breathing detection.
    Buffers frames, runs the trained model, and extracts breathing info.
    """

    def __init__(self, buffer_size=CSI_BUFFER_SIZE):
        self._buffers = {i: collections.deque(maxlen=buffer_size) for i in [1, 2, 3]}
        self._buffer_size = buffer_size

        # Per-node results
        self._ml_prob = {1: 0.0, 2: 0.0, 3: 0.0}
        self._ml_detected = {1: False, 2: False, 3: False}
        self._breathing = {i: {"detected": False, "bpm": 0.0, "band_power": 0.0} for i in [1, 2, 3]}

        # Smoothing (EMA)
        self._smooth_prob = {1: 0.0, 2: 0.0, 3: 0.0}
        self._ema_alpha = 0.3

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
                print(f"[CSI-ML] Model loaded: {model_path}")
            else:
                print(f"[CSI-ML] WARNING: model.pkl or scaler.pkl not found")
                print(f"[CSI-ML]   Run: python train_csi_model.py")
        except Exception as e:
            print(f"[CSI-ML] Load failed: {e}")

    def push_frame(self, node_id: int, amplitude: np.ndarray):
        """Add one CSI amplitude vector (n_subcarriers,) to buffer."""
        if node_id not in self._buffers:
            return
        self._buffers[node_id].append(amplitude.astype(np.float32))

        if len(self._buffers[node_id]) >= self._buffer_size:
            self._run_inference(node_id)

    def _run_inference(self, node_id: int):
        """Run ML model + breathing detection on buffered data."""
        buf = list(self._buffers[node_id])
        amp_matrix = np.stack(buf, axis=1)  # (n_sub, buffer_size)

        # ML prediction
        if self._model is not None:
            try:
                features = preprocess_realtime_csi(
                    amp_matrix, self._scaler, window_size=self._buffer_size
                )
                if len(features) > 0:
                    proba = self._model.predict_proba(features)
                    raw_prob = float(np.mean(proba[:, 1]))

                    # EMA smoothing
                    self._smooth_prob[node_id] = (
                        self._ema_alpha * raw_prob +
                        (1 - self._ema_alpha) * self._smooth_prob[node_id]
                    )
                    self._ml_prob[node_id] = self._smooth_prob[node_id]
                    self._ml_detected[node_id] = self._ml_prob[node_id] > 0.55
            except Exception as e:
                print(f"[CSI-ML] Inference error node={node_id} shape={amp_matrix.shape}: {e}")

        # Breathing detection
        try:
            breath = get_breathing_info(amp_matrix)
            self._breathing[node_id] = {
                "detected": breath["detected"],
                "bpm": round(breath["bpm"], 1),
                "band_power": round(breath["band_power"], 4),
            }
        except Exception as e:
            print(f"[CSI-ML] Breathing detection error node={node_id} shape={amp_matrix.shape}: {e}")

        # Sliding window: clear half buffer
        half = self._buffer_size // 2
        for _ in range(half):
            if self._buffers[node_id]:
                self._buffers[node_id].popleft()

    def get_node_prob(self, node_id: int) -> float:
        return self._ml_prob.get(node_id, 0.0)

    def get_node_detected(self, node_id: int) -> bool:
        return self._ml_detected.get(node_id, False)

    def get_breathing(self, node_id: int) -> dict:
        return self._breathing.get(node_id, {"detected": False, "bpm": 0.0, "band_power": 0.0})

    def get_max_prob(self) -> float:
        return max(self._ml_prob.values())

    @property
    def model_loaded(self) -> bool:
        return self._model is not None


# ── Triangle Consensus ────────────────────────────────────────────────────────

def triangle_consensus(csi_node: CSINodeInference) -> dict:
    """
    Apply triangular consensus logic across 3 nodes.

    Rules:
      - ≥2 nodes detect human → INSIDE triangle (confirmed)
      - 1 node detects → EDGE (possible, low confidence)
      - 0 nodes → NO HUMAN

    Also computes weighted position estimate for heatmap pointer.
    """
    detections = {i: csi_node.get_node_detected(i) for i in [1, 2, 3]}
    probs = {i: csi_node.get_node_prob(i) for i in [1, 2, 3]}
    n_detecting = sum(1 for v in detections.values() if v)

    if n_detecting >= MIN_CONSENSUS:
        status = "INSIDE"
        consensus_detected = True
        # Confidence = weighted average of detecting nodes
        detecting_probs = [p for i, p in probs.items() if detections[i]]
        consensus_prob = np.mean(detecting_probs)
    elif n_detecting == 1:
        status = "EDGE"
        consensus_detected = False  # Not confirmed
        consensus_prob = max(probs.values()) * 0.4  # reduced confidence
    else:
        status = "CLEAR"
        consensus_detected = False
        consensus_prob = 0.0

    # Weighted position estimate
    pos_x, pos_y = _compute_position(probs)

    return {
        "consensus_detected": consensus_detected,
        "consensus_prob": float(np.clip(consensus_prob, 0, 1)),
        "consensus_status": status,
        "n_detecting": n_detecting,
        "node_detections": detections,
        "node_probs": probs,
        "pos_x": pos_x,
        "pos_y": pos_y,
    }


def _compute_position(probs: dict) -> tuple:
    """
    Compute estimated human position using weighted average of node positions.
    x = (s1*x1 + s2*x2 + s3*x3) / (s1+s2+s3)
    y = (s1*y1 + s2*y2 + s3*y3) / (s1+s2+s3)
    """
    total_w = 0.0
    wx, wy = 0.0, 0.0
    for nid in [1, 2, 3]:
        w = max(probs.get(nid, 0.0), 0.01)  # minimum weight to avoid division by zero
        nx, ny = NODE_POSITIONS[nid]
        wx += w * nx
        wy += w * ny
        total_w += w
    x = float(np.clip(wx / total_w, 0.0, 1.0))
    y = float(np.clip(wy / total_w, 0.0, 1.0))
    return round(x, 4), round(y, 4)


# ── Heatmap Data Generator ───────────────────────────────────────────────────

def generate_heatmap_data(probs: dict, grid_size: int = 20) -> list:
    """
    Generate heatmap intensity grid based on node detection probabilities.
    Uses inverse-distance weighting from each node position.

    Returns: list of {x, y, intensity} for the heatmap grid.
    """
    if grid_size < 1:
        raise ValueError(f"grid_size must be >= 1, got {grid_size}")

    denom = grid_size - 1 if grid_size > 1 else 1
    points = []
    for gx in range(grid_size):
        for gy in range(grid_size):
            x = gx / denom
            y = gy / denom
            intensity = 0.0
            for nid in [1, 2, 3]:
                nx, ny = NODE_POSITIONS[nid]
                dist = max(0.05, np.sqrt((x - nx) ** 2 + (y - ny) ** 2))
                intensity += probs.get(nid, 0.0) / dist
            intensity = float(np.clip(intensity / 3.0, 0, 1))
            points.append({"x": round(x, 3), "y": round(y, 3), "v": round(intensity, 4)})
    return points


# ── Globals ───────────────────────────────────────────────────────────────────

detector          = TriangularPresenceDetector(meta_path="model_triangular_meta.json")
parser            = CSIFrameParser()
csi_node          = CSINodeInference()
connected_clients = set()

# Position smoother (moving average)
_pos_history = collections.deque(maxlen=10)

# Per-node latest vitals
_node_vitals = {
    1: {"motion": 0.0, "breathing": 0.0},
    2: {"motion": 0.0, "breathing": 0.0},
    3: {"motion": 0.0, "breathing": 0.0},
}


# ── UDP Receiver ──────────────────────────────────────────────────────────────

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
            if nid in [1, 2, 3]:
                _node_vitals[nid]["motion"] = result.motion_energy
                _node_vitals[nid]["breathing"] = result.breathing_bpm
                detector.update_node(
                    node_id=nid, rssi=-70,
                    motion=result.motion_energy,
                    breathing=result.breathing_bpm,
                )

        elif isinstance(result, CSIFrame):
            nid = result.node_id
            if nid in [1, 2, 3]:
                # Feed to CSI-ML buffer
                csi_node.push_frame(nid, result.amplitude)

                detector.update_node(
                    node_id=nid, rssi=result.rssi,
                    motion=_node_vitals[nid]["motion"],
                    breathing=_node_vitals[nid]["breathing"],
                )


# ── WebSocket Broadcast ──────────────────────────────────────────────────────

async def broadcast_loop():
    global connected_clients
    interval = 1.0 / BROADCAST_HZ
    heatmap_counter = 0

    while True:
        await asyncio.sleep(interval)
        if not connected_clients:
            continue

        tri_result = detector.compute()

        # CSI-ML triangle consensus
        consensus = triangle_consensus(csi_node)
        csi_prob = consensus["consensus_prob"]
        tri_prob = tri_result["ml_prob"]

        # Blend probabilities
        if csi_node.model_loaded:
            combined_prob = (CSI_ML_WEIGHT * csi_prob) + (TRI_ML_WEIGHT * tri_prob)
        else:
            combined_prob = tri_prob

        combined_detected = consensus["consensus_detected"] if csi_node.model_loaded else tri_result["ml_detected"]

        # Smooth position
        raw_pos = (consensus["pos_x"], consensus["pos_y"])
        _pos_history.append(raw_pos)
        smooth_x = float(np.mean([p[0] for p in _pos_history]))
        smooth_y = float(np.mean([p[1] for p in _pos_history]))

        # Log
        if combined_prob > 0.3:
            status = consensus["consensus_status"] if csi_node.model_loaded else "TRI"
            print(f"[AI] {status} | Prob:{combined_prob*100:.0f}% | "
                  f"Nodes:{consensus['n_detecting']}/3 | "
                  f"Pos:({smooth_x:.2f},{smooth_y:.2f})")

        # Per-node data
        node_data = []
        for i in [1, 2, 3]:
            breath = csi_node.get_breathing(i)
            node_data.append({
                "node_id":      i,
                "rssi":         tri_result["node_rssi"].get(i, -100),
                "motion":       tri_result["node_motion"].get(i, 0.0),
                "breathing_bpm": breath["bpm"],
                "breathing_detected": breath["detected"],
                "csi_ml_prob":  round(csi_node.get_node_prob(i), 4),
                "csi_detected": csi_node.get_node_detected(i),
                "active":       i in tri_result["active_node_ids"],
                "pos_x":        NODE_POSITIONS[i][0],
                "pos_y":        NODE_POSITIONS[i][1],
            })

        # Heatmap data (send every 2 seconds to reduce bandwidth)
        heatmap_counter += 1
        include_heatmap = (heatmap_counter % (BROADCAST_HZ * 2) == 0)
        heatmap = generate_heatmap_data(consensus["node_probs"]) if include_heatmap else None

        payload = {
            "type":              "rescue_ai",
            "ai_detected":       combined_detected,
            "ai_prob":           round(combined_prob, 4),
            "ai_mode":           "csi_rf+breathing+triangular" if csi_node.model_loaded else tri_result["ml_mode"],
            "consensus_status":  consensus["consensus_status"],
            "n_detecting":       consensus["n_detecting"],
            "csi_ml_prob":       round(csi_prob, 4),
            "tri_prob":          round(tri_prob, 4),
            "active_nodes":      tri_result["active_nodes"],
            "pos_x":             round(smooth_x, 4),
            "pos_y":             round(smooth_y, 4),
            "nodes":             node_data,
            # Legacy compatibility
            "raw_rssi":          tri_result["node_rssi"].get(1, -100),
            "raw_motion":        tri_result["node_motion"].get(1, 0.0),
            "raw_br":            tri_result["node_breathing"].get(1, 0.0),
            "ai_depth":          0.0,
            "ai_zone":           1 if combined_detected else 0,
        }

        if heatmap is not None:
            payload["heatmap"] = heatmap

        msg = json.dumps(payload)
        dead = set()
        for client in list(connected_clients):
            try:
                await client.send(msg)
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
    print("=" * 60)
    print("  Rescue Backend — CSI Human Detection System")
    print("=" * 60)
    print(f"  UDP:       0.0.0.0:{UDP_PORT} (ESP32 nodes)")
    print(f"  WebSocket: ws://localhost:{WS_PORT} (rescue.html)")
    print(f"  CSI Model: {'LOADED' if csi_node.model_loaded else 'NOT FOUND (run train_csi_model.py)'}")
    print(f"  Blend:     CSI-ML {CSI_ML_WEIGHT*100:.0f}% + Triangular {TRI_ML_WEIGHT*100:.0f}%")
    print(f"  Consensus: {MIN_CONSENSUS}/3 nodes required")
    print(f"  Nodes:     N1{NODE_POSITIONS[1]} N2{NODE_POSITIONS[2]} N3{NODE_POSITIONS[3]}")
    print("=" * 60)

    t = threading.Thread(target=udp_receiver, daemon=True)
    t.start()

    async with serve(ws_handler, "0.0.0.0", WS_PORT):
        print(f"[WS] Server ready. Open rescue.html in browser.\n")
        await broadcast_loop()


if __name__ == "__main__":
    asyncio.run(main())
