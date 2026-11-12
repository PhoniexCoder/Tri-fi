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
| `tests/` | pytest suite: decoder round-trip vs firmware C, firmware host tests, provisioning key/type checks, backend startup |
| `firmware/rescuer_esp8266.ino` | Optional rescuer badge |
| `examples/rescue_backend.py` | Backend: UDP `4444` in → inference/fusion → WebSocket `8002` out |
| `examples/csi_frame_parser.py` | Packet decoder (must match `csi_collector.c` / `edge_processing.h` wire formats) |
| `csi_preprocessing.py` | Feature extraction and FFT breathing estimate |
| `ui/react-rescue/` | React + Vite dashboard (port 3000) |
| `scripts/` | `flash_rescuer.py`, `udp_debug.py`, `mmwave_fusion_bridge.py` |

Models (`*.pth`, `*.pkl`) and `custom_dataset/` are **not in git**; they come from GitHub Releases and sit in the repo root / `examples/`.

## Run

```bash
pip install -r requirements.txt
python -m pytest tests -q                        # all tests (gcc/clang on PATH for firmware ones)
cd examples && python rescue_backend.py          # backend
cd ui/react-rescue && npm install && npm run dev # dashboard → http://localhost:3000
python scripts/udp_debug.py                      # check node packets arrive on 4444
```

Firmware build on Windows needs MSYS env vars stripped (see `firmware/esp32-csi-node/build_firmware.ps1`).
Build both 8 MB (`sdkconfig.defaults.template`) and 4 MB (`sdkconfig.defaults.4mb`) targets after firmware changes.

## Wire protocol

Packet magics (`0xC5110001` raw CSI … `0xC5110006` anchor beacon) are listed in `examples/csi_frame_parser.py`
and must stay unique; `tests/test_csi_frame_parser.py` checks Python against the C headers.
ESP-IDF CSI bytes are (imaginary, real) per subcarrier.

## Known issues (fix in this order)

1. Phase 1 firmware changes (anchor mode, DSP pacing, port 4444) are host-tested only — build with ESP-IDF and verify on 3 nodes + anchor before relying on them.
2. Breathing detrend in `csi_preprocessing.py` acts as a high-pass and removes the 0.1–0.5 Hz band.
3. Consensus accepts 1 node; positions are weighted centroids; extra survivor dots are synthesized.
4. Inference runs on the UDP receive thread; models loaded with `weights_only=False`.
5. Unauthenticated UDP/WebSocket; OTA open unless a PSK is provisioned.
6. Backend assumes node IDs 1–3; the anchor should use node ID 0.

## Rules

- Work on a branch and open a PR into `main`; never commit straight to `main`.
- Never commit secrets, NVS images (`nvs_*.bin`), models, datasets or executables — `.gitignore` blocks them; keep it that way.
- Keep the Python decoder and the C wire structs in sync; any format change updates both plus a round-trip test.
- Don't claim capabilities the code doesn't have (README, dashboard copy, pitch material).
- Read a file before editing it; keep changes scoped to the task.
