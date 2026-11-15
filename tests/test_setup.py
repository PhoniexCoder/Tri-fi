"""Board setup: USB detection, esptool parsing, flash layout and the flashing job (no hardware)."""

import asyncio
import json
import struct
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from websockets.asyncio.client import connect

from trifi import boards
from trifi.api import Api
from trifi.config import Config
from trifi.setup import BoardSetup, build_nvs

PROBE_V5 = """esptool v5.3.1
Connected to ESP32-S3 on COM7:
Chip type:          ESP32-S3 (QFN56) (revision v0.2)
Features:           Wi-Fi, BT 5 (LE), Dual Core + LP Core, 240MHz, Embedded PSRAM 8MB
MAC:                f4:12:fa:aa:bb:cc
Detected flash size: 8MB
"""
PROBE_V4 = "Chip is ESP32-S3 (QFN56) (revision v0.2)\nMAC: F4:12:FA:00:11:22\nDetected flash size: 4MB\n"


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=30))


def fake_port(device, vid, pid=0x1001):
    return SimpleNamespace(device=device, vid=vid, pid=pid, description="USB device", serial_number="X1")


def app_image(version="v1.2.3", project="esp32-csi-node"):
    img = bytearray(0x100)
    img[0] = 0xE9
    struct.pack_into("<I", img, 0x20, boards.APP_DESC_MAGIC)
    img[0x30:0x30 + len(version)] = version.encode()
    img[0x50:0x50 + len(project)] = project.encode()
    img[0x80:0x8B] = b"Nov 15 2026"
    return bytes(img)


@pytest.fixture
def fw_dir(tmp_path):
    for layout in boards.IMAGE_LAYOUT.values():
        for _, name in layout:
            (tmp_path / name).write_bytes(app_image() if name.startswith("esp32-csi-node") else b"\0" * 64)
    return tmp_path


def test_scan_keeps_only_esp32_adapters():
    ports = [fake_port("COM3", 0x8087), fake_port("COM9", 0x1A86),
             fake_port("COM7", 0x303A), fake_port("COM1", None)]
    found = boards.scan(lambda: ports)
    assert [p.device for p in found] == ["COM7", "COM9"]
    assert found[0].adapter == "ESP32 native USB"


@pytest.mark.parametrize("text,chip,mb,mac", [
    (PROBE_V5, "ESP32-S3", 8, "f4:12:fa:aa:bb:cc"),
    (PROBE_V4, "ESP32-S3", 4, "f4:12:fa:00:11:22"),
    ("garbage", None, None, None),
])
def test_parse_probe(text, chip, mb, mac):
    assert boards.parse_probe(text) == {"chip": chip, "flash_mb": mb, "mac": mac}


def test_friendly_errors():
    assert "in use" in boards.friendly_error("could not open port 'COM7': PermissionError", "COM7")
    assert "BOOT" in boards.friendly_error("A fatal error occurred: Failed to connect to ESP32-S3", "COM7")
    assert boards.friendly_error("weird\nlast line", "COM7").endswith("last line")


def test_app_desc_roundtrip_and_rejects_non_images():
    desc = boards.parse_app_desc(app_image("v9"))
    assert desc and desc["version"] == "v9" and desc["project"] == "esp32-csi-node"
    assert boards.parse_app_desc(b"\xff" * 0x100) is None


def test_image_layout_matches_partition_tables():
    for mb in (4, 8):
        offsets = dict((name, off) for off, name in boards.IMAGE_LAYOUT[mb])
        assert offsets["bootloader.bin"] == 0x0 and offsets["ota_data_initial.bin"] == 0xF000
        assert [off for off, n in boards.IMAGE_LAYOUT[mb] if n.startswith("esp32-csi-node")] == [0x20000]
    assert boards.image_files(Path("x"), 16)[-1][1].name == "esp32-csi-node.bin"
    assert boards.image_files(Path("x"), 4)[-1][1].name == "esp32-csi-node-4mb.bin"


def test_find_images_reports_missing(tmp_path, fw_dir):
    assert set(boards.find_images(fw_dir)["sets"]) == {"4", "8"}
    (fw_dir / "partition-table-4mb.bin").unlink()
    found = boards.find_images(fw_dir)
    assert set(found["sets"]) == {"8"} and found["missing"] == ["4 MB: partition-table-4mb.bin"]


def test_progress_is_monotonic_across_images():
    p = boards.FlashProgress([(0x0, 100), (0x100, 100)])
    assert p.feed("Writing at 0x00000032 [==   ] 50%") == pytest.approx(25)
    assert p.feed("Writing at 0x00000150...") == pytest.approx(90)
    assert p.feed("Writing at 0x00000010") == pytest.approx(90)


GOOD = {"port": "COM7", "ssid": "KIT", "password": "secret123", "role": "receiver", "node_id": 2,
        "target_ip": "192.168.137.1", "target_port": 4444, "mode": "full"}


@pytest.mark.parametrize("change,error", [
    ({"port": "COM99"}, "no longer connected"),
    ({"ssid": ""}, "Wi-Fi name"),
    ({"ssid": "x" * 33}, "Wi-Fi name"),
    ({"password": "short"}, "8-63"),
    ({"ssid": "a\nb"}, "control"),
    ({"role": "admin"}, "role"),
    ({"node_id": 0}, "1-255"),
    ({"node_id": True}, "1-255"),
    ({"target_ip": "laptop"}, "IPv4"),
    ({"target_port": 0}, "1-65535"),
    ({"mode": "erase"}, "mode"),
])
def test_validate_flash_rejects(change, error):
    req, err = boards.validate_flash({**GOOD, **change}, ["COM7"])
    assert req is None and error in err


def test_validate_flash_accepts_anchor_and_open_network():
    anchor = {**GOOD, "role": "anchor", "node_id": 0, "password": "", "target_ip": ""}
    req, err = boards.validate_flash(anchor, ["COM7"])
    assert err is None and req.role == "anchor" and req.target_ip == ""


def test_build_nvs_uses_provision_and_handles_dash_password():
    req, _ = boards.validate_flash({**GOOD, "password": "-starts-with-dash"}, ["COM7"])
    nvs = build_nvs(req)
    assert len(nvs) == boards.NVS_SIZE
    assert b"-starts-with-dash" in nvs and b"192.168.137.1" in nvs


class FakeEsptool:
    """Records esptool calls and answers like a connected 8 MB ESP32-S3."""

    def __init__(self, fail_write=False):
        self.calls = []
        self.fail_write = fail_write

    async def __call__(self, args, on_line):
        self.calls.append(args)
        if "flash-id" in args:
            return 0, PROBE_V5
        if "read-flash" in args:
            Path(args[-1]).write_bytes(app_image("v0.9"))
            return 0, "Read 256 bytes"
        if self.fail_write:
            return 2, "A fatal error occurred: Failed to connect to ESP32-S3"
        for line in ("Writing at 0x00009000", "Writing at 0x00020000 [=   ] 10%", "Writing at 0x00020040"):
            on_line(line)
        return 0, "Hash of data verified."


def make_setup(fw_dir, runner, sent):
    cfg = replace(Config(), firmware_dir=fw_dir)
    s = BoardSetup(cfg, broadcast=sent.append, runner=runner,
                   scanner=lambda: boards.scan(lambda: [fake_port("COM7", 0x303A)]),
                   nvs_builder=lambda req: b"\xff" * boards.NVS_SIZE)
    s.missing = None
    s.rescan()
    return s


def test_probe_reads_chip_flash_and_firmware(fw_dir):
    sent = []
    s = make_setup(fw_dir, FakeEsptool(), sent)
    result = run(s.probe("COM7"))
    assert result["state"] == "ok" and result["flash_mb"] == 8 and result["firmware"]["version"] == "v0.9"
    assert sent[-1]["type"] == "setup_state" and sent[-1]["schema"] == 1 and not sent[-1]["busy"]


def test_full_flash_writes_every_image_at_its_offset(fw_dir):
    sent, esp = [], FakeEsptool()
    s = make_setup(fw_dir, esp, sent)
    req, _ = boards.validate_flash(GOOD, s.ports)
    job = run(s.flash(req))
    assert job["state"] == "done" and job["percent"] == 100.0
    write = esp.calls[-1]
    assert "write-flash" in write and "--flash-size" in write
    rest = write[write.index("detect") + 1:]
    pairs = dict(zip(rest[::2], rest[1::2], strict=True))
    assert set(pairs) == {"0x0", "0x8000", "0x9000", "0xf000", "0x20000"}
    assert pairs["0x20000"].endswith("esp32-csi-node.bin")
    assert s.flashed[-1]["node_id"] == 2
    assert all("secret123" not in json.dumps(m) for m in sent)   # password never broadcast


def test_settings_only_writes_just_nvs(fw_dir):
    esp = FakeEsptool()
    s = make_setup(fw_dir, esp, [])
    req, _ = boards.validate_flash({**GOOD, "mode": "settings"}, s.ports)
    assert run(s.flash(req))["state"] == "done"
    write = esp.calls[-1]
    assert write[write.index("write-flash") + 1:][::2] == ["0x9000"]


def test_failed_write_reports_boot_button_hint(fw_dir):
    s = make_setup(fw_dir, FakeEsptool(fail_write=True), [])
    req, _ = boards.validate_flash(GOOD, s.ports)
    job = run(s.flash(req))
    assert job["state"] == "error" and "BOOT" in job["error"] and not s._busy


def test_probe_requests_while_busy_are_queued(fw_dir):
    s = make_setup(fw_dir, FakeEsptool(), [])
    s._scan = lambda: boards.scan(lambda: [fake_port("COM7", 0x303A), fake_port("COM9", 0x1A86)])
    s.rescan()

    async def scenario():
        s._busy = True                       # another board is being checked
        assert s.handle({"type": "setup_probe", "port": "COM9"}) is None
        assert s.probes["COM9"]["state"] == "queued"
        s._idle()                            # that board finished
        for _ in range(50):
            await asyncio.sleep(0.01)
            if s.probes["COM9"].get("state") == "ok" and not s._busy:
                break

    run(scenario())
    assert s.probes["COM9"]["state"] == "ok" and not s._probe_queue


def test_unplugged_board_is_forgotten(fw_dir):
    s = make_setup(fw_dir, FakeEsptool(), [])
    s._scan = lambda: []
    assert s.rescan() and s.state()["ports"] == []


def test_api_refuses_setup_from_remote_clients():
    api = Api(Config(), hello=lambda: {})
    api.setup = SimpleNamespace(handle=lambda data: None)
    local = SimpleNamespace(remote_address=("127.0.0.1", 5000))
    remote = SimpleNamespace(remote_address=("192.168.137.20", 5000))
    msg = json.dumps({"type": "setup_probe", "port": "COM7"})
    assert api._handle_message(msg, local) is None
    assert "laptop" in api._handle_message(msg, remote)
    assert "live" in Api(Config(), hello=lambda: {})._handle_message(msg, local)


def test_live_hello_includes_setup_state(unused_port=18731):
    from trifi.app import LiveApp

    async def scenario():
        cfg = replace(Config(), ws_host="127.0.0.1", ws_port=unused_port,
                      udp_host="127.0.0.1", udp_port=unused_port + 1)
        app = LiveApp(cfg)
        app.setup._scan = lambda: []
        stop = asyncio.Event()
        task = asyncio.create_task(app.run(stop))
        await asyncio.sleep(0.3)
        async with connect(f"ws://127.0.0.1:{unused_port}") as ws:
            hello = json.loads(await ws.recv())
        stop.set()
        await task
        return hello

    hello = run(scenario())
    assert "ports" in hello["setup"] and hello["setup"]["target_port"] == 18732
