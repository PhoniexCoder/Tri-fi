# Changelog

All notable changes to TRI-FI are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Setup view: ESP32 boards plugged into the laptop are detected automatically (USB vendor ID, polled every 1.5 s) with a banner on any view. Each board is checked (chip, flash size, MAC, installed firmware version) and installed with just the Wi-Fi name and password: the 8 or 4 MB image, next free node ID, laptop address and port are filled in. Full install or settings-only, live progress, plain-language errors (port in use, BOOT button), and a "node is sending data" confirmation. Laptop-local clients only; the password is never logged, broadcast or recorded. Backend: `trifi/boards.py`, `trifi/setup.py`, `[setup] firmware_dir`.
- Dashboard v3, rebuilt from scratch with six views (keys 1–6): Overview (status banner, confidence figure, key-number tiles, map, readiness, node agreement, recent events), Map (layer toggles, heatmap on a relative scale, inside-node-area zone, possible-person areas, rescuer badge and confirmed-rescue marker, location tables), Nodes (one card per node with every reported field, plus a comparison table), Signals (nine time-series charts for confidence by source, per-node probability, counts, RSSI, frame rate, loss, breathing, motion and display confidence; 30 s to 10 min range, pause, CSV export), Events (type filters, search, CSV export) and System (connection metrics, model and modes, thresholds, calibration, legacy fields, raw JSON, history export). Charts have crosshair tooltips, keyboard stepping, legends and a table view, and use a colour-blind-checked palette in both themes.
- The hello message now also carries `node_threshold` and `broadcast_hz`.
- Dashboard redesign for field use: one plain-language status banner with next step, confidence vs threshold, nodes agreeing and detection duration; system-readiness checklist ("Ready to search"); 60 s confidence chart with detection periods; event log; one card per node (health, rate, loss, signal, calibration, model, breathing); map with health-coloured nodes and an approximate survivor area; light/dark theme; layouts for desktop, tablet and phone; offline system fonts; data-freshness indicator.
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
- Breathing detection never worked: a 10-sample moving-average detrend high-passed the signal at several Hz, removing the whole 0.1–0.5 Hz band, and the 400-frame window was only 4 s at 100 Hz. It also reported a rate (15–30/min) on pure noise. Now: 30 s window by event time, ~10 Hz block averaging, linear + 10 s drift removal, whitened spectrum, and a peak that must be 12× the noise median and the strongest in 0.05–2 Hz (simulated: ~2% false alarms on noise, none on body sway). The rate is reported only when a rhythm is detected. `breathing_window_frames` default 400 → 3000; replay golden regenerated (only `br1` changed).
- Dashboard dropped every outgoing message (mode changes) in development builds: React's double mount let the first socket's close handler clear the live connection.
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
- README rewritten to describe the current prototype, its known limitations, and a working quick start.
- CLAUDE.md rewritten as a TRI-FI project guide.
- `.gitignore` now blocks NVS images, model/dataset files and executables.
- `requirements.txt` trimmed to the packages the code actually imports.
- `scratch/udp_debug.py` moved to `scripts/udp_debug.py`.

### Removed
- Leftovers from the upstream wifi-densepose project that referenced code not in this repository: `verify`, `install.sh`, `deploy.sh`, `Makefile`, `pyproject.toml`, `example.env`, `.gitmodules`, `.mcp.json`, QEMU helper scripts, and the upstream changelog entries.
- Committed NVS images containing WiFi credentials, and large binaries (models, dataset, `arduino-cli.exe`). Git history was rewritten to purge them; models and data are distributed via GitHub Releases.
