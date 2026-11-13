# Changelog

All notable changes to TRI-FI are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `trifi` backend package replacing `examples/rescue_backend.py`: `python -m trifi run [--record]`, `replay SESSION [--speed]`, `check`.
- Validated TOML kit config (`config/kit.example.toml`): ports, API token, node positions, thresholds, model and session paths.
- Session recording (gzip JSONL of raw packets, mode and detection changes) and deterministic replay, live to the dashboard or in tests.
- Dashboard WebSocket token (`?token=`), schema version on every message, hello message, validated `set_op_mode`, 4 KB message cap.
- Dashboard v2: link-health panel (rate, loss, RSSI, calibration progress), raw model output with the threshold marked, replay badge, schema check.
- Snapshots include per-node link stats, calibration progress, raw confidence and model info.
- CI: ruff, mypy, pytest, dashboard build, ESP-IDF v5.4 firmware builds (8 MB and 4 MB).
- Replay regression test with a golden file; live intake load test at 3 × 100 Hz with slow inference.
- Anchor mode: one node broadcasts ESP-NOW beacons at a fixed rate (default 100 Hz, 6 Mbps OFDM); receivers lock their CSI filter to the first anchor they hear. Provision with `provision.py --role anchor` (`--anchor-hz`, `--no-anchor-lock`).
- Per-node CSI rate, sequence loss and RSSI logged by the backend every 10 s.
- pytest suite (`tests/`): decoder round-trip against the real firmware serializer and structs, packet-magic registry, firmware host tests (anchor, DSP pacer), provisioning key/type checks, backend startup regression.

### Fixed
- Vitals packets were decoded at the wrong offsets and types; fused CSI+mmWave vitals were dropped entirely.
- Raw CSI decoding now honours the header's subcarrier/antenna counts, rejects truncated frames and reads ESP-IDF's (imaginary, real) byte order.
- WASM output packets shared magic `0xC5110004` with fused vitals; moved to `0xC5110005`.
- Firmware default UDP port was 5005 while the backend listens on 4444; default is now 4444.
- On-chip breathing/heart-rate filters assume 20 Hz but received every CSI callback (often 100–500 Hz); input is now paced to 20 Hz.
- Dashboard showed "LINK STABLE" after a disconnect; node markers were drawn at positions that did not match the backend's.
- On a non-UTF-8 Windows console the backend silently ran without its model (a log line's emoji raised inside the load `try`).

### Changed
- Inference runs in a separate worker process, never on the packet-intake path; at most one request in flight per node.
- Models load with `torch.load(weights_only=True)` from `models/`; the legacy RSSI detector no longer unpickles its unused scikit-learn files.
- WebSocket server moved to the current `websockets.asyncio` API.
- Firmware fetches the official wasm3 v0.5.0 release with a SHA-256 check (was a moving third-party fork branch).
- Pinned `requirements.txt`; `requirements-dev.txt` for tests and lint; Python 3.11+.
- `firmware/esp32-csi-node/provision.py` is the only provisioning script; WASM options merged in from the removed `scripts/provision.py`.

### Removed
- Leftovers from the upstream wifi-densepose project that referenced code not in this repository: `verify`, `install.sh`, `deploy.sh`, `Makefile`, `pyproject.toml`, `example.env`, `.gitmodules`, `.mcp.json`, QEMU helper scripts, and the upstream changelog entries.
- Committed NVS images containing WiFi credentials, and large binaries (models, dataset, `arduino-cli.exe`). Git history was rewritten to purge them; models and data are distributed via GitHub Releases.

### Changed
- README rewritten to describe the current prototype, its known limitations, and a working quick start.
- CLAUDE.md rewritten as a TRI-FI project guide.
- `.gitignore` now blocks NVS images, model/dataset files and executables.
- `requirements.txt` trimmed to the packages the code actually imports.
- `scratch/udp_debug.py` moved to `scripts/udp_debug.py`.
