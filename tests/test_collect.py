"""Recording summaries, labels, the data-collection service and dataset export."""

import asyncio
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from synthetic import T0, write_session

from trifi import labels
from trifi.__main__ import main
from trifi.api import Api
from trifi.collect import DataCollection
from trifi.config import Config
from trifi.dataset import export_dataset
from trifi.events import Recorder, session_meta

NAME = "session-20261115-101500.jsonl.gz"


@pytest.fixture(scope="module")
def recorded(tmp_path_factory):
    """A 40 s synthetic session (3 nodes, 100 Hz; breathing from 20 s) under a real session name."""
    d = tmp_path_factory.mktemp("sessions")
    return write_session(d / NAME, Config())


def cfg_for(path: Path) -> Config:
    return replace(Config(), session_dir=path.parent)


@pytest.mark.parametrize("data,error", [
    ({"label": "maybe"}, "label"),
    ({"label": "empty", "people": 2}, "0 people"),
    ({"label": "present", "people": 0}, "at least 1"),
    ({"label": "present", "people": 99}, "people"),
    ({"label": "present", "people": True}, "people"),
    ({"label": "present", "activity": "dancing"}, "activity"),
    ({"label": "present", "setup": "x" * 201}, "setup"),
    ({"label": "present", "setup": "line\nbreak"}, "setup"),
])
def test_label_fields_are_validated(data, error):
    fields, err = labels.validate_fields(data)
    assert fields is None and error in err


def test_label_defaults():
    assert labels.validate_fields({"label": "empty"})[0]["people"] == 0
    assert labels.validate_fields({"label": "present", "setup": " behind brick wall "})[0] == {
        "label": "present", "people": 1, "activity": "", "setup": "behind brick wall"}


def test_ranges_must_fit_the_recording():
    assert labels.validate_range({"start": 5, "end": 4}, 0, 10)[0] is None
    assert labels.validate_range({"start": 0, "end": 50}, 0, 10)[0] is None
    assert labels.validate_range({"start": 1, "end": 9}, 0, 10)[0] == (1.0, 9.0)


def test_label_file_roundtrip_and_close_open(tmp_path):
    s = tmp_path / NAME
    s.write_bytes(b"")
    lf = labels.LabelFile(s)
    seg = lf.add(10.0, None, {"label": "present", "people": 1, "activity": "still", "setup": ""})
    lf.close_open(25.0)
    again = labels.LabelFile(s)
    assert again.find(seg.id).end == 25.0
    assert labels.labels_path(s).name == "session-20261115-101500.labels.json"


@pytest.mark.parametrize("name", ["../" + NAME, "..\\" + NAME, "/etc/passwd", "session-1.jsonl.gz",
                                  42, NAME + "x"])
def test_resolve_session_rejects_anything_but_a_bare_session_name(tmp_path, name):
    (tmp_path / NAME).write_bytes(b"")
    assert labels.resolve_session(tmp_path, name) is None


def test_recorder_writes_a_summary(tmp_path):
    rec = Recorder(tmp_path / NAME, Config())
    rec.packet(100.0, "10.0.0.1:5", b"\x01")
    rec.packet(101.5, "10.0.0.2:5", b"\x02")
    rec.close()
    meta = json.loads(labels.meta_path(tmp_path / NAME).read_text())
    assert meta["packets"] == 2 and meta["first_t"] == 100.0 and meta["last_t"] == 101.5
    assert meta["sources"] == 2 and meta["complete"]


def test_session_meta_scans_once_when_the_summary_is_missing(recorded):
    labels.meta_path(recorded).unlink(missing_ok=True)
    meta = session_meta(recorded)
    assert meta["packets"] > 11_000 and meta["first_t"] == pytest.approx(T0, abs=1)
    assert labels.meta_path(recorded).exists()


def test_live_labelling(tmp_path):
    sent = []
    rec = Recorder(tmp_path / NAME, Config())
    rec.packet(1000.0, "10.0.0.1:5", b"x")
    dc = DataCollection(replace(Config(), session_dir=tmp_path), rec, broadcast=sent.append)
    start = {"type": "label_start", "label": "present", "people": 2, "activity": "breathing"}
    assert dc.handle(start) is None
    assert dc.state()["recording"]["open_segment"]["people"] == 2
    rec.packet(1012.0, "10.0.0.1:5", b"x")
    assert dc.handle({"type": "label_start", "label": "empty"}) is None     # switches: closes the first
    rec.packet(1020.0, "10.0.0.1:5", b"x")
    assert dc.handle({"type": "label_stop"}) is None
    segs = labels.LabelFile(tmp_path / NAME).segments
    assert [(s.label, s.start, s.end) for s in segs] == [("present", 1000.0, 1012.0),
                                                         ("empty", 1012.0, 1020.0)]
    assert sent[-1]["type"] == "data_state" and sent[-1]["schema"] == 1
    assert dc.handle({"type": "label_start", "label": "present"}) is None
    assert "at least 1" in dc.handle({"type": "label_start", "label": "present", "people": 0})
    assert dc.state()["recording"]["open_segment"]["label"] == "present"     # a rejected start keeps it open


def test_live_labelling_needs_a_recording(tmp_path):
    dc = DataCollection(replace(Config(), session_dir=tmp_path), None, broadcast=lambda m: None)
    assert "start a recording first" in dc.handle({"type": "label_start", "label": "empty"})


def test_record_button_starts_and_stops_a_recording(tmp_path):
    sent = []
    dc = DataCollection(replace(Config(), session_dir=tmp_path), None, broadcast=sent.append)
    assert dc.handle({"type": "record_stop"}) == "not recording"
    assert dc.handle({"type": "record_start"}) is None
    assert dc.handle({"type": "record_start"}) == "already recording"
    path = dc.recorder.path
    assert sent[-1]["recording"]["active"] and sent[-1]["recording"]["session"] == path.name
    dc.recorder.packet(500.0, "10.0.0.1:5", b"x")
    assert dc.handle({"type": "label_start", "label": "empty"}) is None
    dc.recorder.packet(530.0, "10.0.0.1:5", b"x")
    assert dc.handle({"type": "record_stop"}) is None             # also closes the open label
    assert dc.recorder is None and path.exists()
    assert json.loads(labels.meta_path(path).read_text())["complete"]
    assert labels.LabelFile(path).segments[0].end == 530.0
    state = dc.state()
    assert not state["recording"]["active"] and state["sessions"][0]["name"] == path.name


def test_a_recording_with_no_packets_is_discarded(tmp_path):
    dc = DataCollection(replace(Config(), session_dir=tmp_path), None, broadcast=lambda m: None)
    dc.handle({"type": "record_start"})
    path = dc.recorder.path
    dc.handle({"type": "record_stop"})
    assert not path.exists() and not labels.meta_path(path).exists()
    assert dc.state()["sessions"] == []


def test_live_app_records_only_while_the_button_is_on(tmp_path):
    from synthetic import raw_packet

    from trifi.app import LiveApp

    app = LiveApp(replace(Config(), session_dir=tmp_path))
    app.setup._scan = lambda: []
    pkt = raw_packet(1, 1, np.full(64, 20.0), -50)
    app.handle_packet(pkt, ("10.0.0.1", 5))
    assert app.recorder is None
    app.data.handle({"type": "record_start"})
    app.handle_packet(pkt, ("10.0.0.1", 5))
    app.handle_packet(pkt, ("10.0.0.1", 5))
    assert app.recorder.packets == 2
    app.data.close()                                               # shutdown finishes the file
    assert app.recorder is None and len(labels.list_sessions(tmp_path)) == 1


def test_label_after_recording(recorded):
    dc = DataCollection(cfg_for(recorded), None, broadcast=lambda m: None)
    dc.rescan()
    assert dc.state()["sessions"][0]["name"] == NAME
    base = {"session": NAME, "type": "label_add"}
    assert dc.handle({**base, "label": "empty", "start": T0 + 1, "end": T0 + 19}) is None
    assert "inside" in dc.handle({**base, "label": "empty", "start": T0 - 100, "end": T0})
    sneaky = {**base, "session": "../" + NAME, "label": "empty", "start": 1, "end": 2}
    assert "unknown recording" in dc.handle(sneaky)
    seg = labels.LabelFile(recorded).segments[0]
    assert dc.handle({"type": "label_update", "session": NAME, "id": seg.id, "setup": "door closed"}) is None
    assert labels.LabelFile(recorded).find(seg.id).setup == "door closed"
    row = dc.state()["sessions"][0]
    assert row["labelled"]["empty"] == pytest.approx(18.0) and row["segments"][0]["setup"] == "door closed"
    assert dc.handle({"type": "label_delete", "session": NAME, "id": seg.id}) is None
    assert labels.LabelFile(recorded).segments == []


def test_export_writes_training_files_and_manifest(recorded, tmp_path):
    lf = labels.LabelFile(recorded)
    lf.segments = []
    lf.add(T0 + 2, T0 + 18, {"label": "empty", "people": 0, "activity": "", "setup": "room"})
    lf.add(T0 + 22, T0 + 38, {"label": "present", "people": 1, "activity": "breathing", "setup": ""})
    lf.add(T0 + 39, None, {"label": "present", "people": 1, "activity": "", "setup": ""})   # open: skipped
    out = tmp_path / "ds"
    stats = export_dataset([recorded], out)
    assert stats["segments"] == 2 and stats["files"] == 6           # 2 ranges × 3 nodes
    a = np.load(next(out.glob("node1_label0_*.npy")))
    assert a.shape[1] == 64 and a.dtype == np.float32 and 1500 <= len(a) <= 1700   # ~16 s at 100 Hz
    rows = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8")))
    assert {r["class"] for r in rows} == {"0", "1"} and {r["node"] for r in rows} == {"1", "2", "3"}
    lf.segments = []
    lf.save()


def test_export_cli(recorded, tmp_path, capsys):
    lf = labels.LabelFile(recorded)
    lf.add(T0 + 2, T0 + 10, {"label": "empty", "people": 0, "activity": "", "setup": ""})
    assert main(["export", "--out", str(tmp_path / "x"), str(recorded)]) == 0
    assert "Exported 3 files" in capsys.readouterr().out
    lf.segments = []
    lf.save()


def test_api_routes_label_messages():
    msg = json.dumps({"type": "label_stop"})
    assert "only available while running live" in Api(Config(), hello=lambda: {})._handle_message(msg)
    api = Api(Config(), hello=lambda: {})
    api.data = type("D", (), {"handle": staticmethod(lambda data: None)})()
    assert api._handle_message(msg) is None


def test_refresh_rescans_in_the_background(recorded):
    sent = []
    dc = DataCollection(cfg_for(recorded), None, broadcast=sent.append)

    async def go():
        assert dc.handle({"type": "data_refresh"}) is None
        for _ in range(100):
            await asyncio.sleep(0.02)
            if sent:
                return

    asyncio.run(go())
    assert sent and sent[-1]["sessions"][0]["name"] == NAME
