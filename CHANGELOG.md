# Changelog

All notable changes to TRI-FI are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Removed
- Leftovers from the upstream wifi-densepose project that referenced code not in this repository: `verify`, `install.sh`, `deploy.sh`, `Makefile`, `pyproject.toml`, `example.env`, `.gitmodules`, `.mcp.json`, QEMU helper scripts, and the upstream changelog entries.
- Committed NVS images containing WiFi credentials, and large binaries (models, dataset, `arduino-cli.exe`). Git history was rewritten to purge them; models and data are distributed via GitHub Releases.

### Changed
- README rewritten to describe the current prototype, its known limitations, and a working quick start.
- CLAUDE.md rewritten as a TRI-FI project guide.
- `.gitignore` now blocks NVS images, model/dataset files and executables.
- `requirements.txt` trimmed to the packages the code actually imports.
- `scratch/udp_debug.py` moved to `scripts/udp_debug.py`.
