"""
Board setup service: watches USB for ESP32 boards and flashes TRI-FI firmware
plus Wi-Fi settings on request from the dashboard.

The dashboard sends:
  {"type": "setup_probe", "port": "COM7"}      read chip, flash size, firmware
  {"type": "setup_flash", "port": ..., "ssid": ..., "password": ..., "role": ...,
   "node_id": ..., "target_ip": ..., "target_port": ..., "mode": "full"|"settings"}

and receives {"type": "setup_state", ...} whenever anything changes. The Wi-Fi
password is passed to the NVS generator only; it is never logged, broadcast
or recorded.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Optional

from . import boards
from .boards import FlashRequest
from .config import REPO_ROOT, Config
from .pipeline import SCHEMA_VERSION

log = logging.getLogger(__name__)

SCAN_INTERVAL_S = 1.5
PROGRESS_BROADCAST_S = 0.25
BAUD = 460800
PROVISION_PY = REPO_ROOT / "firmware" / "esp32-csi-node" / "provision.py"

LineSink = Callable[[str], None]
Runner = Callable[[list[str], LineSink], Awaitable[tuple[int, str]]]


async def run_esptool(args: list[str], on_line: LineSink) -> tuple[int, str]:
    """Run `python -m esptool ARGS`, streaming output lines (split on \\r and \\n)."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "esptool", *args,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
    assert proc.stdout is not None
    out, buf = [], ""
    while True:
        chunk = await proc.stdout.read(256)
        if not chunk:
            break
        buf += chunk.decode("utf-8", "replace")
        *lines, buf = buf.replace("\r", "\n").split("\n")
        for ln in lines:
            if ln.strip():
                out.append(ln)
                on_line(ln)
    if buf.strip():
        out.append(buf)
        on_line(buf)
    return await proc.wait(), "\n".join(out)


def _load_provision() -> Any:
    spec = importlib.util.spec_from_file_location("trifi_provision", PROVISION_PY)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PROVISION_PY}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_nvs(req: FlashRequest) -> bytes:
    """NVS partition image for the request, built and validated by provision.py."""
    prov = _load_provision()
    argv = ["--port", req.port, f"--ssid={req.ssid}", f"--password={req.password}",
            "--role", req.role, "--node-id", str(req.node_id)]
    if req.role == "receiver":
        argv += ["--target-ip", req.target_ip, "--target-port", str(req.target_port)]
    parser = prov.build_parser()
    args = parser.parse_args(argv)
    try:
        prov.validate_args(parser, args)
    except SystemExit as e:
        raise ValueError("settings rejected by provision.py") from e
    return bytes(prov.generate_nvs_binary(prov.build_nvs_csv(args), boards.NVS_SIZE))


def esptool_available() -> Optional[str]:
    """None when esptool and the NVS generator import, else what to install."""
    missing = [pkg for mod, pkg in (("esptool", "esptool"), ("serial", "pyserial"),
                                    ("esp_idf_nvs_partition_gen", "esp-idf-nvs-partition-gen"))
               if importlib.util.find_spec(mod) is None]
    return f"pip install {' '.join(missing)}" if missing else None


class BoardSetup:
    def __init__(self, cfg: Config, broadcast: Callable[[dict[str, Any]], None],
                 runner: Runner = run_esptool,
                 scanner: Callable[[], list[boards.UsbPort]] = boards.scan,
                 nvs_builder: Callable[[FlashRequest], bytes] = build_nvs):
        self.cfg = cfg
        self._broadcast = broadcast
        self._run = runner
        self._scan = scanner
        self._build_nvs = nvs_builder
        self.ports: dict[str, boards.UsbPort] = {}
        self.probes: dict[str, dict[str, Any]] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.flashed: list[dict[str, Any]] = []
        self.missing = esptool_available()
        self.images = boards.find_images(cfg.firmware_dir)
        self.hosts = boards.host_ips()
        self._busy = False
        self._probe_queue: list[str] = []
        self._last_push = 0.0

    # ── state ─────────────────────────────────────────────────────────────────

    def state(self) -> dict[str, Any]:
        return {
            "available": self.missing is None,
            "reason": self.missing,
            "ports": [{**boards.port_dict(u),
                       "probe": self.probes.get(d, {"state": "none"}),
                       "job": self.jobs.get(d, {"state": "idle"})}
                      for d, u in self.ports.items()],
            "images": self.images,
            "host_ips": self.hosts,
            "target_port": self.cfg.udp_port,
            "busy": self._busy,
            "flashed": self.flashed[-20:],
        }

    def push(self, force: bool = True) -> None:
        now = time.monotonic()
        if not force and now - self._last_push < PROGRESS_BROADCAST_S:
            return
        self._last_push = now
        self._broadcast({"type": "setup_state", "schema": SCHEMA_VERSION, **self.state()})

    # ── port watching ─────────────────────────────────────────────────────────

    def rescan(self) -> bool:
        """Refresh the port list; returns True when it changed."""
        try:
            found = {u.device: u for u in self._scan()}
        except Exception:   # a flaky USB stack must not stop the backend
            log.exception("USB port scan failed")
            return False
        if found.keys() == self.ports.keys():
            return False
        for gone in self.ports.keys() - found.keys():
            if self.jobs.get(gone, {}).get("state") != "running":
                self.probes.pop(gone, None)
                self.jobs.pop(gone, None)
        for new in found.keys() - self.ports.keys():
            log.info("ESP32 board connected on %s (%s)", new, found[new].adapter)
        self.ports = found
        return True

    async def run(self, stop: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            changed = await loop.run_in_executor(None, self.rescan)
            if changed:
                self.hosts = boards.host_ips()
                self.images = boards.find_images(self.cfg.firmware_dir)
                self.push()
            try:
                await asyncio.wait_for(stop.wait(), timeout=SCAN_INTERVAL_S)
            except TimeoutError:
                pass

    # ── requests ──────────────────────────────────────────────────────────────

    def handle(self, data: dict[str, Any]) -> Optional[str]:
        if self.missing:
            return f"board setup needs extra tools on this laptop: {self.missing}"
        kind = data.get("type")
        port = data.get("port")
        if kind == "setup_probe":
            if port not in self.ports:
                return "that board is no longer connected"
            if self._busy:   # checked as soon as the current board is done
                if port not in self._probe_queue:
                    self._probe_queue.append(port)
                    self.probes[port] = {"state": "queued"}
                    self.push()
                return None
            asyncio.ensure_future(self.probe(port))
            return None
        if kind == "setup_flash":
            req, error = boards.validate_flash(data, self.ports)
            if error or req is None:
                return error
            if self._busy:
                return "wait for the current board to finish"
            asyncio.ensure_future(self.flash(req))
            return None
        return f"unknown message type {kind!r}"

    async def probe(self, port: str) -> dict[str, Any]:
        self._busy = True
        self.probes[port] = {"state": "probing"}
        self.push()
        try:
            code, text = await self._run(["--port", port, "flash-id"], lambda _: None)
            result: dict[str, Any]
            if code != 0:
                result = {"state": "error", "error": boards.friendly_error(text, port)}
            else:
                info = boards.parse_probe(text)
                result = {"state": "ok", **info, "firmware": None}
                if info["chip"] and info["chip"] != "ESP32-S3":
                    result = {"state": "error", **info,
                              "error": f"This is an {info['chip']}. TRI-FI nodes need an ESP32-S3."}
                elif (info["flash_mb"] or 0) < 4:
                    result = {"state": "error", **info, "error": "This board has less than 4 MB of flash."}
                else:
                    result["firmware"] = await self._read_firmware(port)
            self.probes[port] = result
            return result
        finally:
            self._idle()

    async def _read_firmware(self, port: str) -> Optional[dict[str, str]]:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "app.bin"
            args = ["--port", port, "read-flash", hex(boards.APP_OFFSET), "0x100", str(out)]
            code, _ = await self._run(args, lambda _: None)
            if code != 0 or not out.is_file():
                return None
            return boards.parse_app_desc(out.read_bytes())

    async def flash(self, req: FlashRequest) -> dict[str, Any]:
        self._busy = True
        job: dict[str, Any] = {"state": "running", "stage": "Preparing settings", "percent": 0.0,
                               "node_id": req.node_id, "role": req.role, "mode": req.mode}
        self.jobs[req.port] = job
        self.push()
        log.info("Setting up %s as %s node %d (%s, SSID set, target %s:%d)", req.port, req.role,
                 req.node_id, req.mode, req.target_ip or "-", req.target_port)
        try:
            loop = asyncio.get_running_loop()
            try:
                nvs = await loop.run_in_executor(None, self._build_nvs, req)
            except Exception as e:
                return self._fail(job, f"Couldn't build the settings image: {e}")
            with tempfile.TemporaryDirectory() as tmp:
                nvs_path = Path(tmp) / "nvs.bin"
                nvs_path.write_bytes(nvs)
                files = [(boards.NVS_OFFSET, nvs_path)]
                if req.mode == "full":
                    flash_mb = self.probes.get(req.port, {}).get("flash_mb")
                    if not flash_mb:
                        job["stage"] = "Checking the board"
                        self.push()
                        probe = await self.probe(req.port)
                        self._busy = True
                        flash_mb = probe.get("flash_mb")
                        if probe.get("state") != "ok" or not flash_mb:
                            why = probe.get("error", "Couldn't read the board's flash size.")
                            return self._fail(job, why)
                    images = boards.image_files(self.cfg.firmware_dir, flash_mb)
                    missing = [p.name for _, p in images if not p.is_file()]
                    if missing:
                        return self._fail(job, f"Firmware files missing in {self.cfg.firmware_dir}: "
                                               f"{', '.join(missing)}")
                    files += images
                files.sort()
                progress = boards.FlashProgress([(off, p.stat().st_size) for off, p in files])
                job["stage"] = "Writing firmware and settings" if req.mode == "full" else "Writing settings"

                def on_line(line: str) -> None:
                    pct = progress.feed(line)
                    if pct != job["percent"]:
                        job["percent"] = round(pct, 1)
                        self.push(force=False)

                args = ["--chip", "esp32s3", "--port", req.port, "--baud", str(BAUD), "write-flash"]
                if req.mode == "full":
                    args += ["--flash-size", "detect"]
                for off, path in files:
                    args += [hex(off), str(path)]
                code, text = await self._run(args, on_line)
            if code != 0:
                return self._fail(job, boards.friendly_error(text, req.port))
            job.update(state="done", stage="Done", percent=100.0)
            self.flashed.append({"node_id": req.node_id, "role": req.role, "port": req.port,
                                 "mode": req.mode, "at": time.time()})
            log.info("%s set up as %s node %d", req.port, req.role, req.node_id)
            return job
        finally:
            self._idle()

    def _idle(self) -> None:
        """Mark the service free and start the next queued board check, if any."""
        self._busy = False
        while self._probe_queue:
            port = self._probe_queue.pop(0)
            if port in self.ports:
                asyncio.ensure_future(self.probe(port))
                self._busy = True
                break
        self.push()

    @staticmethod
    def _fail(job: dict[str, Any], message: str) -> dict[str, Any]:
        job.update(state="error", stage="Failed", error=message)
        log.warning("Board setup failed: %s", message)
        return job
