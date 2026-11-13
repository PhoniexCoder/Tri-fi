"""
Deployment configuration.

One TOML file per kit or site. Every key is optional; anything not given keeps
the default below. Unknown keys and out-of-range values are rejected so a typo
cannot silently fall back to a default. Relative paths resolve against the
directory of the config file (or the repository root for built-in defaults).

    [network]
    udp_port = 4444
    ws_port = 8002
    api_token = "change-me"

    [nodes]
    1 = [0.0, 0.0]
    2 = [1.0, 0.0]
    3 = [0.5, 1.0]
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

OP_MODES = {0: "AI (ResNet18)", 1: "Legacy (Triangular)", 2: "Safety Blend (50/50)"}


class ConfigError(ValueError):
    """Invalid configuration file or value."""


@dataclass(frozen=True)
class Config:
    # [network]
    udp_host: str = "0.0.0.0"
    udp_port: int = 4444
    ws_host: str = "0.0.0.0"
    ws_port: int = 8002
    api_token: str = ""                 # empty = no authentication (logged as a warning)

    # [nodes]  node_id -> (x, y) on the unit map
    node_positions: dict[int, tuple[float, float]] = field(
        default_factory=lambda: {1: (0.0, 0.0), 2: (1.0, 0.0), 3: (0.5, 1.0)}
    )

    # [detection]
    op_mode: int = 0                    # see OP_MODES
    window_frames: int = 100            # CSI frames per inference window
    breathing_window_frames: int = 400  # frames for the breathing FFT
    node_detect_threshold: float = 0.45
    ui_detect_threshold: float = 0.30
    broadcast_hz: float = 5.0

    # [models]
    model_dir: Path = REPO_ROOT / "models"
    presence_model: str = "custom_presence_model.pth"   # 2-class, preferred
    fallback_model: str = "model.pth"                   # 6-class

    # [recording]
    record: bool = False
    session_dir: Path = REPO_ROOT / "sessions"

    @property
    def node_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.node_positions))

    def model_candidates(self) -> list[Path]:
        return [self.model_dir / self.presence_model, self.model_dir / self.fallback_model]


# section -> key -> (Config attribute, expected type)
_SCHEMA: dict[str, dict[str, tuple[str, type]]] = {
    "network": {
        "udp_host": ("udp_host", str), "udp_port": ("udp_port", int),
        "ws_host": ("ws_host", str), "ws_port": ("ws_port", int),
        "api_token": ("api_token", str),
    },
    "detection": {
        "op_mode": ("op_mode", int), "window_frames": ("window_frames", int),
        "breathing_window_frames": ("breathing_window_frames", int),
        "node_detect_threshold": ("node_detect_threshold", float),
        "ui_detect_threshold": ("ui_detect_threshold", float),
        "broadcast_hz": ("broadcast_hz", float),
    },
    "models": {
        "dir": ("model_dir", Path), "presence_model": ("presence_model", str),
        "fallback_model": ("fallback_model", str),
    },
    "recording": {"enabled": ("record", bool), "dir": ("session_dir", Path)},
}


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load a TOML config (or defaults when path is None) and validate it."""
    values: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise ConfigError(f"cannot read config {path}: {e}") from e
        values = _parse(data, base=path.resolve().parent)
    values.update(overrides)
    cfg = replace(Config(), **values)
    validate(cfg)
    return cfg


def _parse(data: dict[str, Any], base: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for section, body in data.items():
        if section == "nodes":
            values["node_positions"] = _parse_nodes(body)
            continue
        if section not in _SCHEMA or not isinstance(body, dict):
            raise ConfigError(f"unknown config section [{section}]")
        for key, raw in body.items():
            if key not in _SCHEMA[section]:
                raise ConfigError(f"unknown key '{key}' in [{section}]")
            attr, typ = _SCHEMA[section][key]
            values[attr] = _coerce(raw, typ, f"{section}.{key}", base)
    return values


def _coerce(raw: Any, typ: type, name: str, base: Path) -> Any:
    if typ is Path:
        if not isinstance(raw, str):
            raise ConfigError(f"{name} must be a path string")
        p = Path(raw)
        return p if p.is_absolute() else (base / p).resolve()
    if typ is float and isinstance(raw, int) and not isinstance(raw, bool):
        return float(raw)
    if typ is int and isinstance(raw, bool):
        raise ConfigError(f"{name} must be an integer")
    if not isinstance(raw, typ):
        raise ConfigError(f"{name} must be {typ.__name__}, got {type(raw).__name__}")
    return raw


def _parse_nodes(body: Any) -> dict[int, tuple[float, float]]:
    if not isinstance(body, dict) or not body:
        raise ConfigError("[nodes] must map node IDs to [x, y]")
    nodes: dict[int, tuple[float, float]] = {}
    for key, xy in body.items():
        try:
            nid = int(key)
        except ValueError as e:
            raise ConfigError(f"node ID '{key}' is not an integer") from e
        if (not isinstance(xy, list) or len(xy) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in xy)):
            raise ConfigError(f"node {nid} position must be [x, y]")
        nodes[nid] = (float(xy[0]), float(xy[1]))
    return nodes


def validate(cfg: Config) -> None:
    def check(ok: bool, msg: str) -> None:
        if not ok:
            raise ConfigError(msg)

    for name in ("udp_port", "ws_port"):
        port = getattr(cfg, name)
        check(1 <= port <= 65535, f"{name} must be 1-65535, got {port}")
    check(cfg.udp_port != cfg.ws_port or cfg.udp_host != cfg.ws_host, "udp_port and ws_port must differ")
    check(cfg.op_mode in OP_MODES, f"op_mode must be one of {sorted(OP_MODES)}")
    check(16 <= cfg.window_frames <= 2000, "window_frames must be 16-2000")
    check(cfg.breathing_window_frames >= cfg.window_frames,
          "breathing_window_frames must be >= window_frames")
    for name in ("node_detect_threshold", "ui_detect_threshold"):
        v = getattr(cfg, name)
        check(0.0 <= v <= 1.0, f"{name} must be 0-1, got {v}")
    check(0.5 <= cfg.broadcast_hz <= 50, "broadcast_hz must be 0.5-50")
    check(1 <= len(cfg.node_positions) <= 16, "between 1 and 16 nodes are supported")
    for nid, (x, y) in cfg.node_positions.items():
        check(1 <= nid <= 255, f"node ID {nid} must be 1-255 (0 is reserved for the anchor)")
        check(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, f"node {nid} position must be within [0, 1]")


def describe(cfg: Config) -> dict[str, Any]:
    """Config as plain data for logs and session headers (token redacted)."""
    out: dict[str, Any] = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if f.name == "api_token":
            v = "set" if v else ""
        elif isinstance(v, Path):
            v = str(v)
        elif f.name == "node_positions":
            v = {str(k): list(p) for k, p in v.items()}
        out[f.name] = v
    return out
