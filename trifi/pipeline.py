"""
Deterministic processing core: packets in, dashboard snapshots out.

Ported from examples/rescue_backend.py with the same detection behaviour,
restructured so that:
  - every decision uses the event time passed in (packet receive time), never
    the wall clock, so a recorded session replays to identical output;
  - inference is split into a request (on_packet) and a result
    (on_inference), so the caller decides where the model runs.

Known issues kept on purpose until Phase 4 (see CLAUDE.md): the breathing
detrend, 1-of-N consensus, centroid positions, and synthesised positions for
extra survivors.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import OP_MODES, Config
from .features import get_breathing_info
from .inference import ModelInfo
from .legacy_rssi import TriangularPresenceDetector
from .protocol import CSIFrame, CSIFrameParser, VitalsPacket

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

N_SUBCARRIERS = 64            # model input height; frames are cut or padded to this
CALIB_WINDOWS = 15            # windows of empty-room data for the noise baseline
NOISE_GATE_RATIO = 1.01       # window std must exceed baseline by 1% to pass the gate
GATE_OVERRIDE_CONF = 0.95     # model confidence that bypasses the gate
MIN_BREATHING_FRAMES = 150
DEFAULT_SAMPLE_HZ = 33.0
LINK_RATE_WINDOW_S = 2.0
RESCUER_TIMEOUT_S = 5.0
RESCUER_MESSAGES = {"RESCUER_ACTIVE", "RESCUE_CONFIRMED"}
DISPLAY_FLOOR = 0.40          # probability mapped to 0% on the legacy confidence bar


@dataclass
class InferenceRequest:
    node_id: int
    window: np.ndarray        # normalised (N_SUBCARRIERS, window_frames)
    gated: bool               # noise gate verdict at request time
    t: float


class NodeState:
    def __init__(self, cfg: Config):
        self.amp: collections.deque = collections.deque(maxlen=cfg.window_frames)
        self.breath: collections.deque = collections.deque(maxlen=cfg.breathing_window_frames)
        self.times: collections.deque = collections.deque(maxlen=100)       # sample-rate estimate
        self.arrivals: collections.deque = collections.deque(maxlen=1024)   # link rate
        self.prob_hist: collections.deque = collections.deque(maxlen=5)
        self.survivors_hist: collections.deque = collections.deque(maxlen=10)
        self.calib: list[float] = []
        self.noise_std: Optional[float] = None
        self.ml_prob = 0.0
        self.detected = False
        self.breathing = {"detected": False, "bpm": 0.0, "band_power": 0.0}
        self.motion = 0.0             # from on-chip vitals
        self.vitals_breathing = 0.0   # from on-chip vitals
        self.rssi = -95
        self.pending = False          # an inference request is in flight
        self.dropped_requests = 0

    def survivors(self) -> int:
        if not self.survivors_hist:
            return 0
        hist = list(self.survivors_hist)
        return max(set(hist), key=hist.count)


@dataclass
class RescuerState:
    ip: Optional[str] = None
    active: bool = False
    confirmed: bool = False
    pos_x: float = 0.5
    pos_y: float = 0.5
    last_seen: float = 0.0
    confirmed_pos: Optional[tuple[float, float]] = None


class Pipeline:
    def __init__(self, cfg: Config, start_t: float, model: Optional[ModelInfo] = None):
        self.cfg = cfg
        self.parser = CSIFrameParser()
        self.nodes = {nid: NodeState(cfg) for nid in cfg.node_ids}
        self.legacy = TriangularPresenceDetector(cfg.node_ids, start_t)
        self.model = model
        self.op_mode = cfg.op_mode
        self._pos_history: collections.deque = collections.deque(maxlen=10)
        self._snapshots = 0
        self.rescuer = RescuerState()

    # ── inputs ────────────────────────────────────────────────────────────────

    def set_mode(self, mode: int) -> None:
        if mode not in OP_MODES:
            raise ValueError(f"op_mode must be one of {sorted(OP_MODES)}")
        self.op_mode = mode

    def on_packet(self, data: bytes, src: str, t: float) -> list[InferenceRequest]:
        """Handle one UDP payload received at time t. Returns inference work."""
        if self._handle_rescuer(data, src, t):
            return []

        result = self.parser.parse(data)
        if isinstance(result, VitalsPacket):
            node = self.nodes.get(result.node_id)
            if node is not None:
                node.motion = result.motion_energy
                node.vitals_breathing = result.breathing_bpm
                node.rssi = result.rssi
                self.legacy.update_node(result.node_id, result.rssi, node.motion,
                                        node.vitals_breathing, t)
        elif isinstance(result, CSIFrame):
            node = self.nodes.get(result.node_id)
            if node is not None:
                node.rssi = result.rssi
                node.arrivals.append(t)
                req = self._push_frame(result.node_id, node, result.amplitude, t)
                self.legacy.update_node(result.node_id, result.rssi, node.motion,
                                        node.vitals_breathing, t)
                return [req] if req is not None else []
        return []

    def on_inference(self, req: InferenceRequest, probs: np.ndarray) -> None:
        """Apply a model result for a request created by on_packet."""
        node = self.nodes[req.node_id]
        node.pending = False
        probs = np.asarray(probs, dtype=np.float32)
        binary = len(probs) == 2

        presence_conf = float(probs[1] if binary else 1.0 - probs[0])
        gated = req.gated and presence_conf <= GATE_OVERRIDE_CONF
        if gated:
            probs = np.zeros(len(probs), dtype=np.float32)
            probs[0] = 1.0

        argmax = int(np.argmax(probs))
        node.survivors_hist.append(0 if argmax == 0 else 2 if argmax == 2 else 1)
        node.prob_hist.append(float(1.0 - probs[0]))
        node.ml_prob = float(np.mean(node.prob_hist))
        node.detected = node.ml_prob > self.cfg.node_detect_threshold

    def on_inference_failed(self, req: InferenceRequest) -> None:
        self.nodes[req.node_id].pending = False

    # ── per-node processing ───────────────────────────────────────────────────

    def _push_frame(self, nid: int, node: NodeState, amplitude: np.ndarray,
                    t: float) -> Optional[InferenceRequest]:
        amp = amplitude.astype(np.float32)[:N_SUBCARRIERS]
        if len(amp) < N_SUBCARRIERS:
            amp = np.pad(amp, (0, N_SUBCARRIERS - len(amp)))
        node.amp.append(amp)
        node.breath.append(amp)
        node.times.append(t)

        if len(node.amp) < self.cfg.window_frames:
            return None

        matrix = np.stack(list(node.amp), axis=1)       # (n_sub, window)
        if node.noise_std is None:
            node.calib.append(float(np.std(matrix)))
            if len(node.calib) >= CALIB_WINDOWS:
                node.noise_std = float(np.median(node.calib))
                log.info("Node %d calibrated, baseline std %.3f", nid, node.noise_std)

        req = None
        if self.model is not None:
            if node.pending:
                node.dropped_requests += 1
            else:
                std = float(np.std(matrix)) + 1e-6
                gated = node.noise_std is None or std < node.noise_std * NOISE_GATE_RATIO
                window = ((matrix - float(np.mean(matrix))) / std).astype(np.float32)
                node.pending = True
                req = InferenceRequest(node_id=nid, window=window, gated=gated, t=t)

        self._update_breathing(node)

        for _ in range(self.cfg.window_frames // 2):    # 50% overlap
            node.amp.popleft()
        return req

    @staticmethod
    def _update_breathing(node: NodeState) -> None:
        if len(node.breath) < MIN_BREATHING_FRAMES:
            return
        hz = DEFAULT_SAMPLE_HZ
        times = node.times
        if len(times) > 10 and times[-1] > times[0]:
            hz = (len(times) - 1) / (times[-1] - times[0])
        try:
            b = get_breathing_info(np.stack(list(node.breath), axis=1), sample_rate=hz)
        except (ValueError, FloatingPointError) as e:
            log.warning("Breathing estimate failed: %s", e)
            return
        node.breathing = {"detected": bool(b["detected"]), "bpm": round(b["bpm"], 1),
                          "band_power": round(b["band_power"], 4)}

    # ── rescuer badge ─────────────────────────────────────────────────────────

    def _handle_rescuer(self, data: bytes, src: str, t: float) -> bool:
        if len(data) > 32:
            return False
        try:
            msg = data.decode("utf-8").strip()
        except UnicodeDecodeError:
            return False
        if msg not in RESCUER_MESSAGES:
            return False
        r = self.rescuer
        r.ip, r.active, r.last_seen = src.split(":")[0], True, t
        if msg == "RESCUE_CONFIRMED" and not r.confirmed:
            r.confirmed = True
            r.confirmed_pos = (r.pos_x, r.pos_y)
            log.info("Rescue confirmed at (%s, %s)", r.pos_x, r.pos_y)
        # Known issue (Phase 4): uses each node's last RSSI, not the badge's signal.
        total = wx = wy = 0.0
        for nid, (nx, ny) in self.cfg.node_positions.items():
            w = max(0.01, 100 + self.nodes[nid].rssi)
            wx, wy, total = wx + w * nx, wy + w * ny, total + w
        r.pos_x = round(float(np.clip(wx / total, 0.0, 1.0)), 4)
        r.pos_y = round(float(np.clip(wy / total, 0.0, 1.0)), 4)
        return True

    # ── fusion ────────────────────────────────────────────────────────────────

    def consensus(self) -> dict:
        detections = {i: n.detected for i, n in self.nodes.items()}
        probs = {i: n.ml_prob for i, n in self.nodes.items()}
        n_detecting = sum(detections.values())
        max_prob = max(probs.values()) if probs else 0.0
        if n_detecting >= 3:
            status, detected, prob = "INSIDE", True, max_prob
        elif n_detecting == 2:
            status, detected, prob = "INSIDE", True, max_prob * 0.95
        elif n_detecting == 1:
            status, detected, prob = "EDGE", True, max_prob * 0.85
        else:
            status, detected, prob = "OUT OF RANGE", False, 0.0
        pos_x, pos_y = self._weighted_position(probs)
        return {"consensus_detected": detected, "consensus_prob": float(np.clip(prob, 0, 1)),
                "consensus_status": status, "n_detecting": n_detecting,
                "node_probs": probs, "pos_x": pos_x, "pos_y": pos_y}

    def _weighted_position(self, probs: dict[int, float]) -> tuple[float, float]:
        total = wx = wy = 0.0
        for nid, (nx, ny) in self.cfg.node_positions.items():
            w = max(probs.get(nid, 0.0), 0.01)
            wx, wy, total = wx + w * nx, wy + w * ny, total + w
        return (round(float(np.clip(wx / total, 0.0, 1.0)), 4),
                round(float(np.clip(wy / total, 0.0, 1.0)), 4))

    def heatmap(self, probs: dict[int, float], grid: int = 20) -> list[dict]:
        denom = grid - 1 if grid > 1 else 1
        n = len(self.cfg.node_positions)
        points = []
        for gx in range(grid):
            for gy in range(grid):
                x, y = gx / denom, gy / denom
                v = 0.0
                for nid, (nx, ny) in self.cfg.node_positions.items():
                    v += probs.get(nid, 0.0) / max(0.05, float(np.hypot(x - nx, y - ny)))
                points.append({"x": round(x, 3), "y": round(y, 3),
                               "v": round(float(np.clip(v / n, 0, 1)), 4)})
        return points

    # ── output ────────────────────────────────────────────────────────────────

    def snapshot(self, t: float) -> dict:
        cfg = self.cfg
        model_loaded = self.model is not None
        tri = self.legacy.compute(t)
        cons = self.consensus()
        csi_prob, tri_prob = cons["consensus_prob"], tri["ml_prob"]

        w_csi, w_tri = {0: (1.0, 0.0), 1: (0.0, 1.0)}.get(self.op_mode, (0.5, 0.5))
        combined = w_csi * csi_prob + w_tri * tri_prob if model_loaded else tri_prob
        base_detected = cons["consensus_detected"] if model_loaded else tri["ml_detected"]
        detected = bool(base_detected and combined >= cfg.ui_detect_threshold)

        if model_loaded and detected:
            display = float(np.clip((combined - DISPLAY_FLOOR) / (1 - DISPLAY_FLOOR), 0.0, 1.0))
        elif model_loaded:
            display = 0.0
        else:
            display = combined if detected else 0.0

        self._pos_history.append((cons["pos_x"], cons["pos_y"]))
        sx = float(np.mean([p[0] for p in self._pos_history]))
        sy = float(np.mean([p[1] for p in self._pos_history]))

        survivors = 0
        if detected:
            counts = [n.survivors() for n in self.nodes.values() if n.detected]
            survivors = max(counts) if counts else 0
            survivors = max(survivors, 1)

        survivors_list = []
        if detected:
            if survivors == 1:
                survivors_list.append({"x": round(sx, 4), "y": round(sy, 4),
                                       "intensity": round(combined, 4)})
            else:
                # Known issue (Phase 4): positions for extra survivors are synthesised.
                for i in range(survivors):
                    a = 2 * np.pi * i / survivors
                    survivors_list.append({
                        "x": round(float(np.clip(sx + 0.08 * np.cos(a), 0, 1)), 4),
                        "y": round(float(np.clip(sy + 0.08 * np.sin(a), 0, 1)), 4),
                        "intensity": round(combined * (1.0 - 0.15 * i / survivors), 4)})

        nodes = []
        for nid, n in self.nodes.items():
            px, py = cfg.node_positions[nid]
            nodes.append({
                "node_id": nid, "rssi": tri["node_rssi"].get(nid, -100),
                "motion": round(n.motion, 4), "breathing": n.breathing["bpm"],
                "breathing_detected": n.breathing["detected"],
                "csi_ml_prob": round(n.ml_prob, 4), "csi_detected": n.detected,
                "active": nid in tri["active_node_ids"], "pos_x": px, "pos_y": py,
            })

        r = self.rescuer
        first = cfg.node_ids[0]
        payload = {
            "type": "rescue_ai",
            "schema": SCHEMA_VERSION,
            "t": round(t, 3),
            "ai_detected": detected,
            "ai_prob": round(display, 4),
            "confidence_raw": round(float(combined), 4),
            "ui_threshold": cfg.ui_detect_threshold,
            "ai_mode": OP_MODES[self.op_mode],
            "active_op_mode": self.op_mode,
            "consensus_status": cons["consensus_status"],
            "n_detecting": cons["n_detecting"],
            "csi_ml_prob": round(csi_prob, 4),
            "tri_prob": round(tri_prob, 4),
            "active_nodes": tri["active_nodes"],
            "pos_x": round(sx, 4), "pos_y": round(sy, 4),
            "survivors": survivors,
            "survivors_list": survivors_list,
            "nodes": nodes,
            "links": self.link_stats(t),
            "calibration": {str(nid): {"done": n.noise_std is not None,
                                       "progress": round(min(1.0, len(n.calib) / CALIB_WINDOWS), 3)}
                            for nid, n in self.nodes.items()},
            "model": {"loaded": model_loaded,
                      "classes": self.model.n_classes if self.model else 0},
            "rescuer": {
                "active": r.active and t - r.last_seen < RESCUER_TIMEOUT_S,
                "confirmed": r.confirmed, "x": r.pos_x, "y": r.pos_y,
                "conf_x": r.confirmed_pos[0] if r.confirmed_pos else None,
                "conf_y": r.confirmed_pos[1] if r.confirmed_pos else None,
            },
            # Legacy compatibility fields
            "raw_rssi": tri["node_rssi"].get(first, -100),
            "raw_motion": tri["node_motion"].get(first, 0.0),
            "raw_br": tri["node_breathing"].get(first, 0.0),
            "ai_depth": 0.0,
            "ai_zone": 1 if detected else 0,
        }

        self._snapshots += 1
        if self._snapshots % max(1, int(cfg.broadcast_hz * 2)) == 0:
            payload["heatmap"] = self.heatmap(cons["node_probs"])
        return payload

    def link_stats(self, t: float) -> dict[str, dict]:
        out = {}
        for nid, n in self.nodes.items():
            st = self.parser.stats.get(nid)
            recent = sum(1 for a in n.arrivals if t - a <= LINK_RATE_WINDOW_S)
            out[str(nid)] = {
                "rate_hz": round(recent / LINK_RATE_WINDOW_S, 1),
                "loss": round(st.loss_ratio, 4) if st else 0.0,
                "frames": st.frames if st else 0,
                "rssi": n.rssi,
                "dropped_inference": n.dropped_requests,
            }
        return out
