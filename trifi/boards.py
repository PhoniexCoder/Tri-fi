"""
ESP32 board helpers for the dashboard's Setup view: USB port detection,
esptool output parsing, firmware image discovery and flash layout.

Everything here is pure (no serial I/O) so it can be unit tested; the async
service that actually runs esptool lives in setup.py.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import struct
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

# USB vendor IDs of ESP32 boards and the USB-UART bridges they ship with.
USB_ADAPTERS = {
    0x303A: "ESP32 native USB",
    0x10C4: "CP210x USB-UART",
    0x1A86: "CH340/CH9102 USB-UART",
    0x0403: "FTDI USB-UART",
}

NVS_OFFSET = 0x9000
NVS_SIZE = 0x6000
APP_OFFSET = 0x20000
APP_DESC_MAGIC = 0xABCD5432

# Flash layout per flash size, taken from the release partition tables.
IMAGE_LAYOUT: dict[int, list[tuple[int, str]]] = {
    8: [(0x0, "bootloader.bin"), (0x8000, "partition-table.bin"),
        (0xF000, "ota_data_initial.bin"), (APP_OFFSET, "esp32-csi-node.bin")],
    4: [(0x0, "bootloader.bin"), (0x8000, "partition-table-4mb.bin"),
        (0xF000, "ota_data_initial.bin"), (APP_OFFSET, "esp32-csi-node-4mb.bin")],
}

SSID_MAX_BYTES = 32
PASSWORD_LEN = (8, 63)    # WPA2-PSK passphrase; empty means an open network


@dataclass(frozen=True)
class UsbPort:
    device: str
    adapter: str
    description: str
    vid: int
    pid: int
    serial: str


def scan(comports: Optional[Callable[[], Iterable[Any]]] = None) -> list[UsbPort]:
    """Serial ports that look like an ESP32 board (by USB vendor ID)."""
    if comports is None:
        from serial.tools import list_ports
        comports = list_ports.comports
    found = []
    for p in comports():
        vid = getattr(p, "vid", None)
        if vid not in USB_ADAPTERS:
            continue
        found.append(UsbPort(device=p.device, adapter=USB_ADAPTERS[vid],
                             description=getattr(p, "description", "") or "",
                             vid=vid, pid=getattr(p, "pid", 0) or 0,
                             serial=getattr(p, "serial_number", "") or ""))
    return sorted(found, key=lambda u: u.device)


def port_dict(u: UsbPort) -> dict[str, Any]:
    return asdict(u)


# ── esptool output ────────────────────────────────────────────────────────────

_CHIP = re.compile(r"(?:Chip type:\s*|Chip is |Detecting chip type\.*\s*)(ESP32[-\w]*)", re.I)
_FLASH = re.compile(r"Detected flash size:\s*(\d+)\s*MB", re.I)
_MAC = re.compile(r"MAC:\s*([0-9a-f]{2}(?::[0-9a-f]{2}){5})", re.I)
_PCT = re.compile(r"(\d{1,3}(?:\.\d+)?)\s?%")
_ADDR = re.compile(r"Writing at 0x([0-9a-f]+)", re.I)


def parse_probe(text: str) -> dict[str, Any]:
    chip = _CHIP.search(text)
    flash = _FLASH.search(text)
    mac = _MAC.search(text)
    return {"chip": chip.group(1).upper() if chip else None,
            "flash_mb": int(flash.group(1)) if flash else None,
            "mac": mac.group(1).lower() if mac else None}


def friendly_error(text: str, port: str) -> str:
    """Turn esptool's failure output into an instruction for the operator."""
    t = text.lower()
    if "could not open port" in t or "access is denied" in t or "permissionerror" in t or "port is busy" in t:
        return (f"{port} is in use by another program (serial monitor, Arduino IDE, another flasher). "
                "Close it and try again.")
    if "failed to connect" in t or "no serial data received" in t or "wrong boot mode" in t:
        return ("Couldn't talk to the board. Hold the BOOT button, tap RESET (or unplug and replug "
                "while holding BOOT), release BOOT, then try again.")
    if ("this chip is" in t and "not esp32-s3" in t) or "unexpected chip" in t:
        return "This board isn't an ESP32-S3. TRI-FI nodes need an ESP32-S3."
    if "no module named esptool" in t:
        return "esptool is not installed on this laptop. Run: pip install esptool"
    if "timed out" in t:
        return "The board stopped responding. Check the USB cable (use a data cable) and try again."
    last = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return f"esptool failed: {last[-1] if last else 'unknown error'}"


class FlashProgress:
    """Overall percentage across several images from esptool's progress lines."""

    def __init__(self, segments: list[tuple[int, int]]):
        self.segments = sorted(segments)              # (offset, size)
        self.total = sum(s for _, s in self.segments) or 1
        self.percent = 0.0

    def feed(self, line: str) -> float:
        m = _ADDR.search(line)
        if m:
            addr = int(m.group(1), 16)
            done = 0
            for off, size in self.segments:
                if addr >= off + size:
                    done += size
                elif addr >= off:
                    done += addr - off
            self.percent = max(self.percent, 100.0 * done / self.total)
        return self.percent


# ── firmware images ───────────────────────────────────────────────────────────

def _cstr(b: bytes) -> str:
    return b.split(b"\0", 1)[0].decode("utf-8", "replace").strip()


def parse_app_desc(image: bytes) -> Optional[dict[str, str]]:
    """Version info from the start of an ESP-IDF app image, or None."""
    if len(image) < 0xB0 or image[0] != 0xE9:
        return None
    if struct.unpack_from("<I", image, 0x20)[0] != APP_DESC_MAGIC:
        return None
    return {"version": _cstr(image[0x30:0x50]), "project": _cstr(image[0x50:0x70]),
            "time": _cstr(image[0x70:0x80]), "date": _cstr(image[0x80:0x90]),
            "idf": _cstr(image[0x90:0xB0])}


def find_images(firmware_dir: Path) -> dict[str, Any]:
    """Which full-install image sets are present, with their app version."""
    out: dict[str, Any] = {"dir": str(firmware_dir), "sets": {}, "missing": []}
    for mb, layout in IMAGE_LAYOUT.items():
        paths = [(off, firmware_dir / name) for off, name in layout]
        missing = [p.name for _, p in paths if not p.is_file()]
        if missing:
            out["missing"] += [f"{mb} MB: {m}" for m in missing]
            continue
        app = paths[-1][1]
        with open(app, "rb") as f:
            desc = parse_app_desc(f.read(0x100))
        out["sets"][str(mb)] = {"files": [p.name for _, p in paths],
                                "size": sum(p.stat().st_size for _, p in paths),
                                "app": desc}
    return out


def image_files(firmware_dir: Path, flash_mb: int) -> list[tuple[int, Path]]:
    """(offset, path) list for a board of flash_mb megabytes."""
    layout = IMAGE_LAYOUT[8 if flash_mb >= 8 else 4]
    return [(off, firmware_dir / name) for off, name in layout]


# ── host network ──────────────────────────────────────────────────────────────

HOTSPOT_IP = "192.168.137.1"   # Windows Mobile Hotspot gateway


def host_ips() -> list[dict[str, str]]:
    """This laptop's IPv4 addresses, best guess first (hotspot, then default route)."""
    ips: list[str] = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.append(str(info[4][0]))
    except OSError:
        pass
    default = None
    try:   # UDP connect sends nothing; it only picks the outgoing interface
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            default = s.getsockname()[0]
            ips.append(default)
    except OSError:
        pass
    seen, out = set(), []
    for ip in ips:
        if ip in seen or ip.startswith("127.") or ip.startswith("169.254."):
            continue
        seen.add(ip)
        label = ("Windows hotspot" if ip == HOTSPOT_IP
                 else "Default network" if ip == default else "Other network")
        out.append({"ip": ip, "label": label})
    rank = {"Windows hotspot": 0, "Default network": 1, "Other network": 2}
    return sorted(out, key=lambda d: rank[d["label"]])


# ── request validation ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FlashRequest:
    port: str
    ssid: str
    password: str
    role: str            # receiver | anchor
    node_id: int
    target_ip: str
    target_port: int
    mode: str            # full | settings


def _int(v: Any) -> Optional[int]:
    return v if isinstance(v, int) and not isinstance(v, bool) else None


def validate_flash(data: dict[str, Any],
                   ports: Iterable[str]) -> tuple[Optional[FlashRequest], Optional[str]]:
    port = data.get("port")
    if port not in set(ports):
        return None, "that board is no longer connected"
    ssid, password = data.get("ssid"), data.get("password", "")
    if not isinstance(ssid, str) or not ssid or len(ssid.encode()) > SSID_MAX_BYTES:
        return None, f"Wi-Fi name must be 1-{SSID_MAX_BYTES} bytes"
    lo_pw, hi_pw = PASSWORD_LEN
    if not isinstance(password, str) or (password and not lo_pw <= len(password) <= hi_pw):
        return None, "Wi-Fi password must be 8-63 characters (or empty for an open network)"
    if any(ord(c) < 32 for c in ssid + password):
        return None, "Wi-Fi name and password can't contain control characters"
    role = data.get("role")
    if role not in ("receiver", "anchor"):
        return None, "role must be receiver or anchor"
    node_id = _int(data.get("node_id"))
    lo = 0 if role == "anchor" else 1
    if node_id is None or not lo <= node_id <= 255:
        return None, f"node ID must be {lo}-255 for a {role}"
    target_ip = data.get("target_ip", "")
    target_port = _int(data.get("target_port"))
    if role == "receiver":
        try:
            if not isinstance(target_ip, str) or ipaddress.ip_address(target_ip).version != 4:
                raise ValueError
        except ValueError:
            return None, "laptop address must be an IPv4 address like 192.168.137.1"
        if target_port is None or not 1 <= target_port <= 65535:
            return None, "laptop port must be 1-65535"
    mode = data.get("mode")
    if mode not in ("full", "settings"):
        return None, "mode must be full or settings"
    return FlashRequest(port=port, ssid=ssid, password=password, role=role, node_id=node_id,
                        target_ip=target_ip if role == "receiver" else "",
                        target_port=target_port or 4444, mode=mode), None
