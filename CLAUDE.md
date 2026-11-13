# TRI-FI — Project Guide

WiFi CSI sensing on ESP32-S3 nodes to detect a breathing person behind walls and debris.
Current scope: **CSI only** (phone detection, acoustic, CO2 and radar are future work).
Status: early prototype heading toward a commercial demo; through-debris detection is not yet validated.

## Layout

| Path | What |
|---|---|
| `firmware/esp32-csi-node/` | ESP-IDF v5.4 firmware: CSI capture (`csi_collector.c`), on-chip DSP (`edge_processing.c`), anchor mode (`anchor.c`), NVS config, OTA, UDP sender |
| `firmware/esp32-csi-node/provision.py` | The only provisioning script (WiFi, target IP/port, node ID, role, MAC filter, TDM, edge, WASM) |
| `firmware/esp32-csi-node/test/` | Host stubs, `wire_dump.c` and `anchor_test.c` used by pytest; libFuzzer targets |
| `firmware/rescuer_esp8266.ino` | Optional rescuer badge |
| `trifi/` | Backend package (`python -m trifi run|replay|check`) |
| `trifi/pipeline.py` | Deterministic core: packets + event time in, snapshots out; no sockets, threads or clock |
| `trifi/protocol.py` | Packet decoder (must match `csi_collector.c` / `edge_processing.h` / `anchor.h`) |
| `trifi/inference.py` | ResNet loading (`weights_only=True`) and the worker process |
| `trifi/app.py`, `trifi/api.py` | asyncio UDP intake, 5 Hz snapshots, WebSocket API (token, schema v1) |
| `trifi/events.py` | Session recording (gzip JSONL of raw packets) and deterministic replay |
| `trifi/config.py` | Validated TOML config; see `config/kit.example.toml` |
| `trifi/features.py`, `trifi/legacy_rssi.py` | Breathing FFT / features; legacy RSSI detector (op modes 1–2) |
| `ui/react-rescue/` | React + Vite dashboard (port 3000); `src/lib/useTrifiSocket.js` speaks the API |
| `tests/` | pytest: protocol round-trip vs firmware C, firmware host tests, provisioning, pipeline, replay golden, API, live intake load |
| `scripts/` | `flash_rescuer.py`, `udp_debug.py`, `mmwave_fusion_bridge.py` |

Models (`models/*.pth`) and `custom_dataset/` are **not in git**; they come from GitHub Releases.

## Run

```bash
pip install -r requirements.txt          # requirements-dev.txt for tests/lint (no torch needed)
python -m trifi check                    # preflight
python -m trifi run [--record]           # live backend
python -m trifi replay sessions/<f>.jsonl.gz
cd ui/react-rescue && npm install && npm run dev   # http://localhost:3000[?token=...]
ruff check trifi tests && mypy && python -m pytest  # gcc/clang on PATH for firmware tests
```

Firmware: CI builds 8 MB (`sdkconfig.defaults.template`) and 4 MB (`sdkconfig.defaults.4mb`) with ESP-IDF v5.4.
Local builds on Windows need MSYS env vars stripped (see `firmware/esp32-csi-node/build_firmware.ps1`).

## Replay regression

`tests/test_replay.py` replays a synthetic session (`tests/synthetic.py`) with a stub predictor and compares
against `tests/data/replay_golden.json`. Any change to detection behaviour changes the golden file: regenerate
with `TRIFI_UPDATE_GOLDEN=1 python -m pytest tests/test_replay.py`, and review the diff as part of the change.

## Wire protocol

Packet magics (`0xC5110001` raw CSI … `0xC5110006` anchor beacon) are listed in `trifi/protocol.py`
and must stay unique; `tests/test_protocol.py` checks Python against the C headers.
ESP-IDF CSI bytes are (imaginary, real) per subcarrier. WebSocket messages carry `"schema": 1`;
bump `SCHEMA_VERSION` in `trifi/pipeline.py` and `ui/react-rescue/src/lib/useTrifiSocket.js` together.

## Known issues (fix in this order)

1. Phase 1 firmware changes (anchor mode, DSP pacing, port 4444) are host-tested and CI-compiled only — verify on 3 receivers + anchor.
2. Breathing detrend in `trifi/features.py` acts as a high-pass and removes the 0.1–0.5 Hz band.
3. Consensus accepts 1 node; positions are weighted centroids; extra survivor dots are synthesised; rescuer position uses node RSSI, not the badge.
4. Node UDP is unauthenticated; OTA open unless a PSK is provisioned.
5. Node IDs come from config (default 1–3); the anchor uses node ID 0 and is not a configured node.

## Rules

- Work on a branch and open a PR into `main`; never commit straight to `main`.
- Never commit secrets, NVS images (`nvs_*.bin`), models, datasets, sessions or executables — `.gitignore` blocks them.
- Keep the Python decoder and the C wire structs in sync; any format change updates both plus a round-trip test.
- `trifi/pipeline.py` must stay deterministic: use the event time passed in, never `time.time()`.
- Don't claim capabilities the code doesn't have (README, dashboard copy, pitch material).
- Read a file before editing it; keep changes scoped to the task.
