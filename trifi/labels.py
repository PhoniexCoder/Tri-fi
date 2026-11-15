"""
Labels and summaries for recorded sessions.

Each session file gets two small JSON sidecars next to it (the raw recording
is never rewritten):

    session-20261115-101500.jsonl.gz
    session-20261115-101500.meta.json     packets, first/last packet time, size
    session-20261115-101500.labels.json   labelled time ranges (segments)

A segment is a time range on the session's own clock (epoch seconds, the
same clock as the packets) with what was in the area during it.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

LABELS = ("empty", "present")                    # exported as class 0 / 1

# The six classes of custom_dataset/ (recorded 4-11 April 2026, file names
# node<N>_label<C>_<time>.npy) and of the 6-class model trained on it:
# class -> (people in the area, what they were doing).
DATASET_CLASSES: dict[int, tuple[int, str]] = {
    0: (0, "empty"),
    1: (1, "one person sitting"),
    2: (1, "one person moving in the centre"),
    3: (1, "one person moving near the nodes (perimeter)"),
    4: (2, "two people sitting or standing"),
    5: (2, "two people walking inside"),
}


def people_for_class(cls: int, n_classes: int) -> int:
    """People count a model's predicted class stands for."""
    if cls == 0:
        return 0
    if n_classes == len(DATASET_CLASSES):
        return DATASET_CLASSES.get(cls, (1, ""))[0]
    return 1            # presence model: 1 = someone is there; it can't count
ACTIVITIES = ("", "still", "breathing", "moving")
MAX_PEOPLE = 20
MAX_SETUP_CHARS = 200
MAX_SEGMENTS = 2000
SESSION_RE = re.compile(r"^session-\d{8}-\d{6}(?:-\d+)?\.jsonl(?:\.gz)?$")
LABELS_VERSION = 1


def session_base(path: Path) -> str:
    name = path.name
    for suffix in (".jsonl.gz", ".jsonl"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def meta_path(session: Path) -> Path:
    return session.with_name(session_base(session) + ".meta.json")


def labels_path(session: Path) -> Path:
    return session.with_name(session_base(session) + ".labels.json")


def write_json(path: Path, data: Any) -> None:
    """Atomic write: a crash never leaves a half-written sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


@dataclass
class Segment:
    id: str
    start: float
    end: Optional[float]          # None while a live segment is still open
    label: str
    people: int
    activity: str = ""
    setup: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def validate_fields(data: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """label / people / activity / setup from a client message."""
    label = data.get("label")
    if label not in LABELS:
        return None, f"label must be one of {', '.join(LABELS)}"
    people = data.get("people", 0 if label == "empty" else 1)
    if not isinstance(people, int) or isinstance(people, bool) or not 0 <= people <= MAX_PEOPLE:
        return None, f"people must be 0-{MAX_PEOPLE}"
    if label == "empty" and people != 0:
        return None, "an empty area has 0 people"
    if label == "present" and people == 0:
        return None, "a person-present label needs at least 1 person"
    activity = data.get("activity", "")
    if activity not in ACTIVITIES:
        return None, f"activity must be one of {', '.join(a for a in ACTIVITIES if a)} or blank"
    setup = data.get("setup", "")
    if not isinstance(setup, str) or len(setup) > MAX_SETUP_CHARS or any(ord(c) < 32 for c in setup):
        return None, f"setup notes must be plain text, at most {MAX_SETUP_CHARS} characters"
    return {"label": label, "people": people, "activity": activity, "setup": setup.strip()}, None


def validate_range(data: dict[str, Any], first: Optional[float],
                   last: Optional[float]) -> tuple[Optional[tuple[float, float]], Optional[str]]:
    start, end = _num(data.get("start")), _num(data.get("end"))
    if start is None or end is None or end <= start:
        return None, "start and end must be numbers with end after start"
    if first is not None and last is not None and (start < first - 1 or end > last + 1):
        return None, "the range must lie inside the recording"
    return (round(start, 3), round(end, 3)), None


class LabelFile:
    """Segments of one session, loaded from and saved to its sidecar."""

    def __init__(self, session: Path):
        self.session = session
        self.path = labels_path(session)
        raw = read_json(self.path) or {}
        self.segments: list[Segment] = []
        for s in raw.get("segments", []) if isinstance(raw, dict) else []:
            try:
                end = None if s.get("end") is None else float(s["end"])
                self.segments.append(Segment(
                    id=str(s["id"]), start=float(s["start"]), end=end, label=str(s["label"]),
                    people=int(s.get("people", 0)), activity=str(s.get("activity", "")),
                    setup=str(s.get("setup", ""))))
            except (KeyError, TypeError, ValueError):
                continue        # skip a hand-edited entry we can't read

    def save(self) -> None:
        self.segments.sort(key=lambda s: s.start)
        write_json(self.path, {"version": LABELS_VERSION, "session": self.session.name,
                               "segments": [s.to_dict() for s in self.segments]})

    def add(self, start: float, end: Optional[float], fields: dict[str, Any]) -> Segment:
        if len(self.segments) >= MAX_SEGMENTS:
            raise ValueError(f"at most {MAX_SEGMENTS} labelled ranges per recording")
        seg = Segment(id=uuid.uuid4().hex[:10], start=start, end=end, **fields)
        self.segments.append(seg)
        self.save()
        return seg

    def find(self, seg_id: Any) -> Optional[Segment]:
        return next((s for s in self.segments if s.id == seg_id), None)

    def close_open(self, t: float) -> None:
        changed = False
        for s in self.segments:
            if s.end is None:
                s.end = max(t, s.start)
                changed = True
        if changed:
            self.save()


def summarize(session: Path, meta: Optional[dict[str, Any]]) -> dict[str, Any]:
    """One row of the dashboard's recordings list."""
    lf = LabelFile(session)
    first = meta.get("first_t") if meta else None
    last = meta.get("last_t") if meta else None
    covered = {"empty": 0.0, "present": 0.0}
    for s in lf.segments:
        end = s.end if s.end is not None else last
        if end is not None:
            covered[s.label] = covered.get(s.label, 0.0) + max(0.0, end - s.start)
    return {
        "name": session.name,
        "size": session.stat().st_size if session.exists() else 0,
        "first_t": first, "last_t": last,
        "duration": (last - first) if first is not None and last is not None else None,
        "packets": meta.get("packets") if meta else None,
        "sources": meta.get("sources") if meta else None,
        "labelled": {k: round(v, 1) for k, v in covered.items()},
        "segments": [s.to_dict() for s in lf.segments],
    }


def list_sessions(session_dir: Path) -> list[Path]:
    if not session_dir.is_dir():
        return []
    return sorted((p for p in session_dir.iterdir() if p.is_file() and SESSION_RE.match(p.name)),
                  key=lambda p: p.name, reverse=True)


def resolve_session(session_dir: Path, name: Any) -> Optional[Path]:
    """A session file by bare name, only inside session_dir (no traversal)."""
    if not isinstance(name, str) or not SESSION_RE.match(name):
        return None
    path = (session_dir / name).resolve()
    if path.parent != session_dir.resolve() or not path.is_file():
        return None
    return path
