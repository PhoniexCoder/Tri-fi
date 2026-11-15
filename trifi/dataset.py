"""
Export labelled ranges of recorded sessions as training files.

For every closed labelled range and every node that sent CSI during it:

    <out>/node<N>_label<C>_<start>_<segment id>.npy     float32 (frames, 64) amplitudes
    <out>/manifest.csv                                  one row per file

C is the class index (0 = empty, 1 = person present), so the files load with
the same "node<N>_label<C>_..." convention as custom_dataset/. People count,
activity and setup notes are in the manifest, not the file name.
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
from .labels import LABELS, LabelFile
from .pipeline import N_SUBCARRIERS
from .protocol import CSIFrame, CSIFrameParser

MIN_FRAMES = 100        # skip node/range pairs with less than ~1 s of data
MANIFEST_FIELDS = ["file", "session", "segment", "node", "label", "class", "people", "activity",
                   "setup", "start", "end", "frames"]


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


def export_dataset(sessions: Iterable[Path], out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    stats = {"sessions": 0, "segments": 0, "files": 0, "frames": 0}
    for session in sessions:
        session = Path(session)
        segments = [s for s in LabelFile(session).segments if s.end is not None and s.label in LABELS]
        if not segments:
            continue
        stats["sessions"] += 1
        nodes = _node_frames(session)
        for seg in segments:
            stats["segments"] += 1
            for nid, (t, a) in sorted(nodes.items()):
                mask = (t >= seg.start) & (t < seg.end)
                n = int(mask.sum())
                if n < MIN_FRAMES:
                    continue
                cls = LABELS.index(seg.label)
                name = f"node{nid}_label{cls}_{int(seg.start)}_{seg.id}.npy"
                np.save(out / name, a[mask])
                rows.append({"file": name, "session": session.name, "segment": seg.id, "node": nid,
                             "label": seg.label, "class": cls, "people": seg.people,
                             "activity": seg.activity, "setup": seg.setup, "start": seg.start,
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
