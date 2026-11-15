"""
Append-only session log and deterministic replay.

A session file is JSON Lines (gzip-compressed when the name ends in .gz):

    {"kind": "session", "version": 1, "started": "...", "config": {...}}
    {"kind": "udp", "t": 1731400000.123, "src": "192.168.137.12:50123", "data": "<base64>"}
    {"kind": "mode", "t": ..., "mode": 2}
    {"kind": "detection", "t": ..., "detected": true, "confidence": 0.71, "status": "INSIDE"}

Raw packets are stored exactly as received, so a session can be re-run
through any later version of the pipeline. "detection" lines are written for
humans and audits; replay ignores them and recomputes.
"""

from __future__ import annotations

import base64
import gzip
import json
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Optional

import numpy as np

from .config import Config, describe
from .labels import meta_path, read_json, write_json
from .pipeline import Pipeline

SESSION_VERSION = 1


def _open(path: Path, mode: str) -> IO[str]:
    if mode not in ("r", "w"):
        raise ValueError(f"mode must be 'r' or 'w', got {mode!r}")
    if path.suffix == ".gz":
        return gzip.open(path, "wt" if mode == "w" else "rt", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


class Recorder:
    """Writes a session file; flushes regularly so a crash loses little."""

    def __init__(self, path: Path, cfg: Config, flush_every: int = 200):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f = _open(path, "w")
        self._pending = 0
        self._flush_every = flush_every
        self.started = datetime.now(UTC).isoformat(timespec="seconds")
        self.packets = 0
        self.first_t: Optional[float] = None
        self.last_t: Optional[float] = None
        self._sources: set[str] = set()
        self._write({"kind": "session", "version": SESSION_VERSION,
                     "started": self.started, "config": describe(cfg)})

    @staticmethod
    def default_path(cfg: Config) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return cfg.session_dir / f"session-{stamp}.jsonl.gz"

    def _write(self, rec: dict[str, Any]) -> None:
        self._f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._pending += 1
        if self._pending >= self._flush_every:
            self._f.flush()
            self._pending = 0

    def packet(self, t: float, src: str, data: bytes) -> None:
        self._write({"kind": "udp", "t": round(t, 6), "src": src,
                     "data": base64.b64encode(data).decode("ascii")})
        self.packets += 1
        if self.first_t is None:
            self.first_t = t
        self.last_t = t
        self._sources.add(src.split(":")[0])

    def mode(self, t: float, mode: int) -> None:
        self._write({"kind": "mode", "t": round(t, 6), "mode": mode})

    def detection(self, t: float, detected: bool, confidence: float, status: str) -> None:
        self._write({"kind": "detection", "t": round(t, 6), "detected": detected,
                     "confidence": confidence, "status": status})

    def meta(self) -> dict[str, Any]:
        return {"session": self.path.name, "started": self.started, "packets": self.packets,
                "first_t": self.first_t, "last_t": self.last_t, "sources": len(self._sources),
                "complete": self._f.closed}

    def write_meta(self) -> None:
        """Flush and refresh the summary sidecar (called periodically and on close)."""
        if not self._f.closed:
            self._f.flush()
        write_json(meta_path(self.path), self.meta())

    def close(self) -> None:
        self._f.close()
        self.write_meta()


@dataclass
class SessionRecord:
    kind: str
    t: float
    src: str = ""
    data: bytes = b""
    mode: int = 0


def read_session(path: Path) -> Iterator[SessionRecord]:
    """Yield udp and mode records in file order."""
    with _open(Path(path), "r") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON") from e
            kind = rec.get("kind")
            if kind == "session":
                if rec.get("version") != SESSION_VERSION:
                    raise ValueError(f"{path}: unsupported session version {rec.get('version')}")
            elif kind == "udp":
                yield SessionRecord("udp", float(rec["t"]), rec.get("src", ""),
                                    base64.b64decode(rec["data"]))
            elif kind == "mode":
                yield SessionRecord("mode", float(rec["t"]), mode=int(rec["mode"]))


def session_meta(path: Path) -> dict[str, Any]:
    """Summary of a session: from its sidecar, else by scanning the file once (then cached)."""
    meta = read_json(meta_path(path))
    if isinstance(meta, dict) and meta.get("complete"):
        return meta
    packets, first, last, sources = 0, None, None, set()
    try:
        for rec in read_session(path):
            if rec.kind != "udp":
                continue
            packets += 1
            first = rec.t if first is None else first
            last = rec.t
            sources.add(rec.src.split(":")[0])
    except (OSError, ValueError, EOFError):
        pass                # a session cut off by a crash: keep what was readable
    meta = {"session": path.name, "packets": packets, "first_t": first, "last_t": last,
            "sources": len(sources), "complete": True}
    try:
        write_json(meta_path(path), meta)
    except OSError:
        pass
    return meta


def replay(path: Path, cfg: Config, predict: Optional[Callable[[np.ndarray], np.ndarray]],
           n_classes: int = 2) -> Iterator[dict]:
    """
    Re-run a session through a fresh pipeline as fast as possible.

    Inference runs inline (synchronously) with `predict`, and snapshots are
    taken every 1/broadcast_hz seconds of session time, so the output depends
    only on the session file, the config and the predictor.
    """
    from .inference import ModelInfo

    records = iter(read_session(path))
    first = next(records, None)
    if first is None:
        return
    model = ModelInfo(path=Path("<replay>"), n_classes=n_classes) if predict else None
    pipe = Pipeline(cfg, start_t=first.t, model=model)
    period = 1.0 / cfg.broadcast_hz
    next_snap = first.t + period

    def handle(rec: SessionRecord) -> None:
        if rec.kind == "mode":
            pipe.set_mode(rec.mode)
            return
        for req in pipe.on_packet(rec.data, rec.src, rec.t):
            assert predict is not None
            pipe.on_inference(req, predict(req.window))

    for rec in _chain(first, records):
        while rec.t >= next_snap:
            yield pipe.snapshot(next_snap)
            next_snap += period
        handle(rec)
    yield pipe.snapshot(next_snap)


def _chain(first: SessionRecord, rest: Iterator[SessionRecord]) -> Iterator[SessionRecord]:
    yield first
    yield from rest
