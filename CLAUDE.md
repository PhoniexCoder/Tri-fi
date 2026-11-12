# TRI-FI — Project Guide

WiFi CSI sensing on ESP32-S3 nodes to detect a breathing person behind walls and debris.
Current scope: **CSI only** (phone detection, acoustic, CO2 and radar are future work).
Status: early prototype heading toward a commercial demo; through-debris detection is not yet validated.

## Layout

| Path | What |
|---|---|
| `firmware/esp32-csi-node/` | ESP-IDF v5.4 firmware: CSI capture (`csi_collector.c`), on-chip DSP (`edge_processing.c`), NVS config, OTA, UDP sender |
| `firmware/esp32-csi-node/provision.py` | Canonical node provisioning (WiFi, target IP/port, node ID, MAC filter, TDM, edge settings) |
| `firmware/rescuer_esp8266.ino` | Optional rescuer badge |
| `examples/rescue_backend.py` | Backend: UDP `4444` in → inference/fusion → WebSocket `8002` out |
| `examples/csi_frame_parser.py` | Packet decoder (must match `csi_collector.c` / `edge_processing.h` wire formats) |
| `csi_preprocessing.py` | Feature extraction and FFT breathing estimate |
| `ui/react-rescue/` | React + Vite dashboard (port 3000) |
| `scripts/` | `provision.py` (older copy), `flash_rescuer.py`, `udp_debug.py`, `mmwave_fusion_bridge.py` |

Models (`*.pth`, `*.pkl`) and `custom_dataset/` are **not in git**; they come from GitHub Releases and sit in the repo root / `examples/`.

## Run

```bash
pip install -r requirements.txt
cd examples && python rescue_backend.py          # backend
cd ui/react-rescue && npm install && npm run dev # dashboard → http://localhost:3000
python scripts/udp_debug.py                      # check node packets arrive on 4444
```

Firmware build on Windows needs MSYS env vars stripped (see `firmware/esp32-csi-node/build_firmware.ps1`).
Build both 8 MB (`sdkconfig.defaults.template`) and 4 MB (`sdkconfig.defaults.4mb`) targets after firmware changes.

## Known issues (fix in this order)

1. Vitals packet decoded at wrong offsets in `csi_frame_parser.py` (firmware layout: `<IBBHIbBxxffII`, 32 bytes); `0xC5110003/0004` packets dropped; `EDGE_FUSED_MAGIC` collides with `WASM_OUTPUT_MAGIC`.
2. Firmware default UDP port 5005 vs backend 4444.
3. No router-free anchor transmit mode (needed for a fixed 100 Hz packet rate).
4. Breathing detrend in `csi_preprocessing.py` acts as a high-pass and removes the 0.1–0.5 Hz band.
5. Consensus accepts 1 node; positions are weighted centroids; extra survivor dots are synthesized.
6. Inference runs on the UDP receive thread; models loaded with `weights_only=False`.
7. Unauthenticated UDP/WebSocket; OTA open unless a PSK is provisioned.
8. On a non-UTF-8 Windows console, the emoji in the model-load `print` raises inside the `try` in `CSINodeInference._load_model`, so the backend silently runs with no model. Workaround: `set PYTHONIOENCODING=utf-8`.

## Rules

- Work on a branch and open a PR into `main`; never commit straight to `main`.
- Never commit secrets, NVS images (`nvs_*.bin`), models, datasets or executables — `.gitignore` blocks them; keep it that way.
- Keep the Python decoder and the C wire structs in sync; any format change updates both plus a round-trip test.
- Don't claim capabilities the code doesn't have (README, dashboard copy, pitch material).
- Read a file before editing it; keep changes scoped to the task.
