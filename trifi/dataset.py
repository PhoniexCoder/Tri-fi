"""
Export labelled ranges of recorded sessions as training files.

For every closed labelled range and every node that sent CSI during it:

    <out>/node<N>_label<C>_<start>_<segment id>.npy     float32 (frames, 64) amplitudes
    <out>/manifest.csv                                  one row per file

With classes=2 (default), C is 0 = empty, 1 = person present. With classes=6,
C is the custom_dataset/ class (see DATASET_CLASSES in labels.py: 0 empty,
1 one sitting, 2 one moving in the centre, 3 one moving near the nodes,
4 two still, 5 two walking), so new recordings combine with the April data;
ranges that don't map to a class are skipped and counted by reason.
People, activity, position and setup notes are in the manifest.
Amplitudes are cut or padded to 64 subcarriers, as in the live pipeline.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .events import read_session
from .labels import LABELS, LabelFile, six_class
from .pipeline import N_SUBCARRIERS
from .protocol import CSIFrame, CSIFrameParser

MIN_FRAMES = 100        # skip node/range pairs with less than ~1 s of data
MANIFEST_FIELDS = ["file", "session", "segment", "node", "label", "class", "scheme", "people", "activity",
                   "place", "setup", "start", "end", "frames"]


def _node_frames(session: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """node -> (times (n,), amplitudes (n, 64)) for every CSI frame in a session."""
    parser = CSIFrameParser()
    times: dict[int, list[float]] = defaultdict(list)
    amps: dict[int, list[np.ndarray]] = defaultdict(list)
    for rec in read_session(session):
        if rec.kind != "udp":
            continue
        frame = parser.parse(rec.data)
        if not isinstance(frame, CSIFrame):
            continue
        a = frame.amplitude.astype(np.float32)[:N_SUBCARRIERS]
        if len(a) < N_SUBCARRIERS:
            a = np.pad(a, (0, N_SUBCARRIERS - len(a)))
        times[frame.node_id].append(rec.t)
        amps[frame.node_id].append(a)
    return {nid: (np.asarray(times[nid]), np.stack(amps[nid])) for nid in times}


def export_dataset(sessions: Iterable[Path], out: Path, classes: int = 2) -> dict[str, Any]:
    if classes not in (2, 6):
        raise ValueError("classes must be 2 or 6")
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"sessions": 0, "segments": 0, "files": 0, "frames": 0, "skipped": {}}
    for session in sessions:
        session = Path(session)
        segments = [s for s in LabelFile(session).segments if s.end is not None and s.label in LABELS]
        if not segments:
            continue
        stats["sessions"] += 1
        nodes = _node_frames(session)
        for seg in segments:
            if classes == 6:
                cls6, why = six_class(seg.label, seg.people, seg.activity, seg.place)
                if cls6 is None:
                    stats["skipped"][why] = stats["skipped"].get(why, 0) + 1
                    continue
            stats["segments"] += 1
            for nid, (t, a) in sorted(nodes.items()):
                mask = (t >= seg.start) & (t < seg.end)
                n = int(mask.sum())
                if n < MIN_FRAMES:
                    continue
                cls = cls6 if classes == 6 else LABELS.index(seg.label)
                name = f"node{nid}_label{cls}_{int(seg.start)}_{seg.id}.npy"
                np.save(out / name, a[mask])
                rows.append({"file": name, "session": session.name, "segment": seg.id, "node": nid,
                             "label": seg.label, "class": cls, "scheme": classes, "people": seg.people,
                             "activity": seg.activity, "place": seg.place, "setup": seg.setup,
                             "start": seg.start,
                             "end": seg.end, "frames": n})
                stats["files"] += 1
                stats["frames"] += n
    manifest = out / "manifest.csv"
    write_header = not manifest.exists()
    with open(manifest, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)
    return stats
