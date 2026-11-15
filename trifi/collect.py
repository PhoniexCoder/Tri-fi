"""
Data collection: the live recording, labelling from the dashboard, and the
list of recorded sessions.

Dashboard -> backend:
  {"type": "label_start", "label": "empty"|"present", "people": n, "activity": "", "setup": ""}
      starts a labelled range now in the live recording (closing any open one)
  {"type": "label_stop"}                       closes the open range
  {"type": "label_add", "session": name, "start": t, "end": t, ...fields}
  {"type": "label_update", "session": name, "id": id, ...fields and/or start/end}
  {"type": "label_delete", "session": name, "id": id}
  {"type": "data_refresh"}                     re-scan the recordings folder

Backend -> dashboard: {"type": "data_state", ...} whenever something changes,
and every few seconds while recording.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

from . import labels
from .config import Config
from .events import Recorder, session_meta
from .pipeline import SCHEMA_VERSION

log = logging.getLogger(__name__)

LIVE_PUSH_S = 2.0
META_WRITE_S = 10.0
MAX_LISTED = 200
LOW_DISK_BYTES = 1 << 30


class DataCollection:
    def __init__(self, cfg: Config, recorder: Optional[Recorder],
                 broadcast: Callable[[dict[str, Any]], None], clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.recorder = recorder
        self._broadcast = broadcast
        self.clock = clock
        self._meta_cache: dict[str, dict[str, Any]] = {}
        self._rows: list[dict[str, Any]] = []
        self._live_labels = labels.LabelFile(recorder.path) if recorder else None

    # ── state ─────────────────────────────────────────────────────────────────

    def _meta(self, path: Path) -> dict[str, Any]:
        if self.recorder and path == self.recorder.path:
            return self.recorder.meta()
        cached = self._meta_cache.get(path.name)
        if cached is None:
            cached = self._meta_cache[path.name] = session_meta(path)
        return cached

    def rescan(self) -> None:
        """Rebuild the recordings list (blocking: scans files without a summary)."""
        rows = []
        for path in labels.list_sessions(self.cfg.session_dir)[:MAX_LISTED]:
            try:
                rows.append({**labels.summarize(path, self._meta(path)),
                             "live": bool(self.recorder and path == self.recorder.path)})
            except OSError:
                continue
        self._rows = rows

    def _refresh_live_row(self) -> None:
        if not self.recorder:
            return
        row = {**labels.summarize(self.recorder.path, self.recorder.meta()), "live": True}
        self._rows = [row] + [r for r in self._rows if r["name"] != row["name"]]

    def state(self) -> dict[str, Any]:
        try:
            free: Optional[int] = shutil.disk_usage(self.cfg.session_dir if self.cfg.session_dir.exists()
                                                    else self.cfg.session_dir.parent).free
        except OSError:
            free = None
        open_seg = None
        if self._live_labels:
            open_seg = next((s.to_dict() for s in self._live_labels.segments if s.end is None), None)
        rec = self.recorder
        return {
            "recording": {"active": rec is not None, "session": rec.path.name if rec else None,
                          "packets": rec.packets if rec else 0,
                          "first_t": rec.first_t if rec else None, "last_t": rec.last_t if rec else None,
                          "open_segment": open_seg},
            "sessions": self._rows,
            "dir": str(self.cfg.session_dir),
            "free_bytes": free,
            "low_disk": free is not None and free < LOW_DISK_BYTES,
            "labels": list(labels.LABELS),
            "activities": [a for a in labels.ACTIVITIES if a],
        }

    def push(self) -> None:
        self._broadcast({"type": "data_state", "schema": SCHEMA_VERSION, **self.state()})

    async def run(self, stop: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.rescan)
        self.push()
        last_meta = loop.time()
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=LIVE_PUSH_S)
            except TimeoutError:
                pass
            if self.recorder:
                if loop.time() - last_meta >= META_WRITE_S:
                    self.recorder.write_meta()
                    last_meta = loop.time()
                self._refresh_live_row()
                self.push()

    def close(self) -> None:
        """Close any open live range at the last packet time (on shutdown)."""
        if self._live_labels and self.recorder:
            self._live_labels.close_open(self.recorder.last_t or self.clock())

    # ── requests ──────────────────────────────────────────────────────────────

    def handle(self, data: dict[str, Any]) -> Optional[str]:
        kind = data.get("type")
        try:
            if kind == "data_refresh":
                asyncio.ensure_future(self._rescan_and_push())
                return None
            if kind in ("label_start", "label_stop"):
                return self._live(kind, data)
            if kind in ("label_add", "label_update", "label_delete"):
                return self._edit(kind, data)
        except (OSError, ValueError) as e:
            return f"could not save labels: {e}"
        return f"unknown message type {kind!r}"

    async def _rescan_and_push(self) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self.rescan)
        self.push()

    def _live(self, kind: str, data: dict[str, Any]) -> Optional[str]:
        if not self.recorder or not self._live_labels:
            return "not recording: start the backend without --no-record to label live"
        now = self.recorder.last_t or self.clock()
        fields: Optional[dict[str, Any]] = None
        if kind == "label_start":            # validate before touching the open range
            fields, error = labels.validate_fields(data)
            if error or fields is None:
                return error
        self._live_labels.close_open(now)
        if fields is not None:
            self._live_labels.add(now, None, fields)
            log.info("Live label started: %s (%d people)", fields["label"], fields["people"])
        else:
            log.info("Live label stopped")
        self._refresh_live_row()
        self.push()
        return None

    def _edit(self, kind: str, data: dict[str, Any]) -> Optional[str]:
        path = labels.resolve_session(self.cfg.session_dir, data.get("session"))
        if path is None:
            return "unknown recording"
        meta = self._meta(path)
        live = self._live_labels
        lf = live if live and path == live.session else labels.LabelFile(path)
        if kind == "label_add":
            fields, error = labels.validate_fields(data)
            rng, rerr = labels.validate_range(data, meta.get("first_t"), meta.get("last_t"))
            if error or rerr or fields is None or rng is None:
                return error or rerr
            lf.add(rng[0], rng[1], fields)
        else:
            seg = lf.find(data.get("id"))
            if seg is None:
                return "that labelled range no longer exists"
            if kind == "label_delete":
                lf.segments.remove(seg)
            else:
                merged = {"label": seg.label, "people": seg.people, "activity": seg.activity,
                          "setup": seg.setup, **{k: data[k] for k in ("label", "people", "activity", "setup")
                                                 if k in data}}
                fields, error = labels.validate_fields(merged)
                if error or fields is None:
                    return error
                if "start" in data or "end" in data:
                    if seg.end is None:
                        return "stop the live range before changing its times"
                    rng, rerr = labels.validate_range({"start": data.get("start", seg.start),
                                                       "end": data.get("end", seg.end)},
                                                      meta.get("first_t"), meta.get("last_t"))
                    if rerr or rng is None:
                        return rerr
                    seg.start, seg.end = rng
                seg.label, seg.people = fields["label"], fields["people"]
                seg.activity, seg.setup = fields["activity"], fields["setup"]
            lf.save()
        self._rows = [({**labels.summarize(path, meta), "live": r.get("live", False)}
                       if r["name"] == path.name else r) for r in self._rows]
        self.push()
        return None
