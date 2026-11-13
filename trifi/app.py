"""
Wires the stages together for live operation and for dashboard replays.

Live:   UDP socket -> Pipeline.on_packet (decode + buffer, never blocks)
                   -> InferenceWorker (separate process) -> Pipeline.on_inference
        timer      -> Pipeline.snapshot -> Api.broadcast
        everything -> Recorder (optional)
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .api import Api
from .config import OP_MODES, Config
from .events import Recorder, replay
from .inference import InferenceWorker
from .pipeline import InferenceRequest, Pipeline

log = logging.getLogger(__name__)

UDP_RCVBUF_BYTES = 1 << 20
LINK_LOG_INTERVAL_S = 10.0


class _UdpProtocol(asyncio.DatagramProtocol):
    def __init__(self, on_datagram: Callable[[bytes, tuple], None]):
        self._on_datagram = on_datagram

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        self._on_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:
        log.warning("UDP error: %s", exc)


def open_udp_socket(host: str, port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, UDP_RCVBUF_BYTES)
    except OSError:
        log.warning("Could not enlarge UDP receive buffer")
    sock.bind((host, port))
    sock.setblocking(False)
    return sock


class LiveApp:
    def __init__(self, cfg: Config, worker: Optional[InferenceWorker] = None,
                 recorder: Optional[Recorder] = None, clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.worker = worker
        self.recorder = recorder
        self.clock = clock
        self.pipeline = Pipeline(cfg, start_t=clock(), model=worker.info if worker else None)
        self.api = Api(cfg, hello=self.hello, on_set_mode=self.set_mode)
        self.packets = 0
        self._inflight: set[asyncio.Task] = set()
        self._last_detected = False

    # ── inputs ────────────────────────────────────────────────────────────────

    def handle_packet(self, data: bytes, addr: tuple) -> None:
        t = self.clock()
        src = f"{addr[0]}:{addr[1]}"
        self.packets += 1
        if self.recorder:
            self.recorder.packet(t, src, data)
        for req in self.pipeline.on_packet(data, src, t):
            task = asyncio.ensure_future(self._infer(req))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)

    async def _infer(self, req: InferenceRequest) -> None:
        assert self.worker is not None
        try:
            probs = await self.worker.predict(req.window)
        except Exception:   # worker crash must not take down intake
            log.exception("Inference failed for node %d", req.node_id)
            self.pipeline.on_inference_failed(req)
            return
        self.pipeline.on_inference(req, probs)

    def set_mode(self, mode: int) -> None:
        self.pipeline.set_mode(mode)
        if self.recorder:
            self.recorder.mode(self.clock(), mode)

    # ── outputs ───────────────────────────────────────────────────────────────

    def hello(self) -> dict[str, Any]:
        return hello_message(self.cfg, self.pipeline, live=True)

    def tick(self) -> dict[str, Any]:
        t = self.clock()
        snap = self.pipeline.snapshot(t)
        if snap["ai_detected"] != self._last_detected:
            self._last_detected = snap["ai_detected"]
            log.info("Detection %s (confidence %.2f, %s)",
                     "ON" if snap["ai_detected"] else "OFF",
                     snap["confidence_raw"], snap["consensus_status"])
            if self.recorder:
                self.recorder.detection(t, snap["ai_detected"], snap["confidence_raw"],
                                        snap["consensus_status"])
        self.api.broadcast(snap)
        return snap

    def log_links(self) -> None:
        links = self.pipeline.link_stats(self.clock())
        parts = [f"N{nid} {s['rate_hz']:5.1f} Hz loss {s['loss'] * 100:4.1f}% rssi {s['rssi']}"
                 for nid, s in links.items()]
        log.info("[LINK] %s | dropped=%d skipped=%d", " | ".join(parts),
                 self.pipeline.parser.n_dropped, self.pipeline.parser.n_skipped)

    # ── run ───────────────────────────────────────────────────────────────────

    async def run(self, stop: Optional[asyncio.Event] = None) -> None:
        loop = asyncio.get_running_loop()
        stop = stop or asyncio.Event()
        sock = open_udp_socket(self.cfg.udp_host, self.cfg.udp_port)
        transport, _ = await loop.create_datagram_endpoint(
            lambda: _UdpProtocol(self.handle_packet), sock=sock)
        server = await self.api.start()
        log.info("Listening: UDP %s:%d, WebSocket %s:%d, model %s, mode %s",
                 self.cfg.udp_host, self.cfg.udp_port, self.cfg.ws_host, self.cfg.ws_port,
                 self.worker.info.path if self.worker else "none (legacy RSSI only)",
                 OP_MODES[self.pipeline.op_mode])
        period = 1.0 / self.cfg.broadcast_hz
        next_link_log = loop.time() + LINK_LOG_INTERVAL_S
        try:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=period)
                except TimeoutError:
                    pass
                self.tick()
                if loop.time() >= next_link_log:
                    self.log_links()
                    next_link_log += LINK_LOG_INTERVAL_S
        finally:
            transport.close()
            server.close()
            await server.wait_closed()
            for task in list(self._inflight):
                task.cancel()
            if self.worker:
                self.worker.close()
            if self.recorder:
                self.recorder.close()
                log.info("Session saved to %s", self.recorder.path)


def hello_message(cfg: Config, pipeline: Optional[Pipeline], live: bool) -> dict[str, Any]:
    return {
        "live": live,
        "node_ids": list(cfg.node_ids),
        "positions": {str(k): list(v) for k, v in cfg.node_positions.items()},
        "op_modes": {str(k): v for k, v in OP_MODES.items()},
        "model": {"loaded": bool(pipeline and pipeline.model),
                  "classes": pipeline.model.n_classes if pipeline and pipeline.model else 0},
        "ui_threshold": cfg.ui_detect_threshold,
    }


async def serve_replay(cfg: Config, session: Path,
                       predict: Optional[Callable[[np.ndarray], np.ndarray]], n_classes: int,
                       speed: float = 1.0, stop: Optional[asyncio.Event] = None) -> int:
    """Replay a session to dashboards at `speed` x real time. Returns snapshot count."""
    stop = stop or asyncio.Event()
    api = Api(cfg, hello=lambda: hello_message(cfg, None, live=False))
    server = await api.start()
    period = 1.0 / cfg.broadcast_hz / speed
    count = 0
    log.info("Replaying %s at %.1fx on ws://%s:%d", session, speed, cfg.ws_host, cfg.ws_port)
    try:
        for snap in replay(session, cfg, predict, n_classes):
            snap["replay"] = True
            api.broadcast(snap)
            count += 1
            try:
                await asyncio.wait_for(stop.wait(), timeout=period)
                break
            except TimeoutError:
                pass
    finally:
        server.close()
        await server.wait_closed()
    log.info("Replay finished: %d snapshots", count)
    return count
