"""
Command line:

    python -m trifi run    [--config FILE] [--record]      live backend (default)
    python -m trifi replay SESSION [--config FILE] [--speed X]
    python -m trifi check  [--config FILE]                 preflight: config + model
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .config import ConfigError, describe, load_config

log = logging.getLogger("trifi")


def _setup_logging(verbose: bool) -> None:
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")   # never crash on console encoding
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


def _load_predictor(cfg):
    """Return (model_path, n_classes, predict) or (None, 0, None)."""
    from .inference import find_model, load_model

    path = find_model(cfg.model_candidates())
    if path is None:
        return None, 0, None
    model = load_model(path)
    return path, model.n_classes, model.predict


def cmd_check(cfg) -> int:
    print(f"TRI-FI {__version__}")
    for key, value in describe(cfg).items():
        print(f"  {key:24} {value}")
    try:
        path, n_classes, _ = _load_predictor(cfg)
    except ImportError as e:
        print(f"  model                    torch not installed ({e})")
        return 1
    except Exception as e:   # corrupt or incompatible weights
        print(f"  model                    FAILED to load: {e}")
        return 1
    if path is None:
        print("  model                    none found (legacy RSSI mode only)")
        return 1
    print(f"  model                    {path} ({n_classes} classes) OK")
    return 0


async def cmd_run(cfg, record: bool) -> None:
    from .app import LiveApp
    from .events import Recorder
    from .inference import InferenceWorker, find_model, probe_model

    worker = None
    path = find_model(cfg.model_candidates())
    if path is None:
        log.warning("No model found in %s; running legacy RSSI detection only", cfg.model_dir)
    else:
        worker = InferenceWorker(probe_model(path))
    recorder = Recorder(Recorder.default_path(cfg), cfg) if record or cfg.record else None
    if recorder:
        log.info("Recording session to %s", recorder.path)
    await LiveApp(cfg, worker=worker, recorder=recorder).run()


async def cmd_replay(cfg, session: Path, speed: float) -> None:
    from .app import serve_replay

    path, n_classes, predict = _load_predictor(cfg)
    if path is None:
        log.warning("No model found; replay uses legacy RSSI detection only")
    await serve_replay(cfg, session, predict, n_classes, speed=speed)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m trifi", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="TOML config file")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="live backend (default)")
    run.add_argument("--record", action="store_true", help="record a session file")
    rep = sub.add_parser("replay", help="replay a recorded session to the dashboard")
    rep.add_argument("session", type=Path)
    rep.add_argument("--speed", type=float, default=1.0, help="playback speed (default 1.0)")
    sub.add_parser("check", help="validate config and model, then exit")
    args = parser.parse_args(argv)

    _setup_logging(args.verbose)
    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        return 2

    if args.command == "check":
        return cmd_check(cfg)
    try:
        if args.command == "replay":
            if not args.session.is_file():
                print(f"Session not found: {args.session}", file=sys.stderr)
                return 2
            if not 0.1 <= args.speed <= 100:
                print("--speed must be 0.1-100", file=sys.stderr)
                return 2
            asyncio.run(cmd_replay(cfg, args.session, args.speed))
        else:
            asyncio.run(cmd_run(cfg, getattr(args, "record", False)))
    except KeyboardInterrupt:
        pass
    except OSError as e:
        print(f"Network error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
