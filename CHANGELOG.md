# Changelog

All notable changes to TRI-FI are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Anchor mode: one node broadcasts ESP-NOW beacons at a fixed rate (default 100 Hz, 6 Mbps OFDM); receivers lock their CSI filter to the first anchor they hear. Provision with `provision.py --role anchor` (`--anchor-hz`, `--no-anchor-lock`).
- Per-node CSI rate, sequence loss and RSSI logged by the backend every 10 s.
- pytest suite (`tests/`): decoder round-trip against the real firmware serializer and structs, packet-magic registry, firmware host tests (anchor, DSP pacer), provisioning key/type checks, backend startup regression.

### Fixed
- Vitals packets were decoded at the wrong offsets and types; fused CSI+mmWave vitals were dropped entirely.
- Raw CSI decoding now honours the header's subcarrier/antenna counts, rejects truncated frames and reads ESP-IDF's (imaginary, real) byte order.
- WASM output packets shared magic `0xC5110004` with fused vitals; moved to `0xC5110005`.
- Firmware default UDP port was 5005 while the backend listens on 4444; default is now 4444.
- On-chip breathing/heart-rate filters assume 20 Hz but received every CSI callback (often 100–500 Hz); input is now paced to 20 Hz.
- On a non-UTF-8 Windows console the backend silently ran without its model (a log line's emoji raised inside the load `try`).

### Changed
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
