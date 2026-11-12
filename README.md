<div align="center">

# 📡 TRI-FI
### WiFi CSI sensing for finding people behind walls and debris

*A low-cost search aid built on ESP32-S3 nodes.*

</div>

---

> **Status: early prototype.** TRI-FI streams WiFi Channel State Information (CSI) from ESP32-S3 nodes to a laptop, runs a presence model and a breathing estimator, and shows the result on a dashboard. Detection through real rubble has **not yet been validated**; that is the next milestone. Treat every output as a lead for conventional confirmation, never as proof that an area is clear.

---

## What it is

When a person breathes, their chest moves a few millimetres. That movement slightly changes how WiFi signals travel through a space. CSI exposes those changes per subcarrier (64 values per packet instead of one signal-strength number), so a node can pick up the slow, rhythmic pattern of breathing (about 6–30 breaths per minute, 0.1–0.5 Hz).

TRI-FI places several ESP32-S3 nodes around a search area, streams their CSI to a laptop, and looks for that pattern.

## How it works today

```
ESP32-S3 nodes ──UDP──► Python backend ──WebSocket──► React dashboard
 CSI capture             ResNet-18 presence model       per-node status,
 on-chip vitals          + FFT breathing estimate       detections, map
                         + multi-node consensus
```

| Part | Location | Notes |
|---|---|---|
| Node firmware (ESP-IDF, C) | `firmware/esp32-csi-node/` | CSI capture, on-chip breathing/heart-rate filters, presence and fall detection, NVS runtime config, OTA |
| Backend (Python) | `examples/rescue_backend.py` | UDP `4444` in, WebSocket `8002` out |
| Frame parser | `examples/csi_frame_parser.py` | Decodes node packets |
| Signal processing | `csi_preprocessing.py` | Features and FFT breathing estimate |
| Dashboard (React + Vite) | `ui/react-rescue/` | Dev server on port `3000` |
| Rescuer badge (ESP8266) | `firmware/rescuer_esp8266.ino` | Optional heartbeat / confirm button |
| Tools | `scripts/` | Provisioning, rescuer flashing, UDP sniffer, mmWave bridge |

### Known limitations (being fixed)

- The vitals packet is decoded at the wrong byte offsets.
- Firmware sends to UDP port `5005` by default; the backend listens on `4444` (provision nodes with `--target-port 4444`).
- The backend's breathing filter suppresses the breathing band; one node is currently enough to report a detection.
- Positions shown are a weighted average of node locations, not true localisation.
- Control channels (UDP, WebSocket, OTA without a PSK) are unauthenticated.

## Quick start

### Requirements

- Python 3.10+ and Node.js 18+
- ESP-IDF v5.4 (only to build firmware)
- 3 or more ESP32-S3 boards (only for live data)

### 1. Get the models

Trained models and the recorded dataset are not stored in git. Download them from the project's GitHub Releases and place them in the repository root:

```
model.pth                     # 6-class ResNet-18 (fallback)
custom_presence_model.pth     # 2-class presence ResNet-18 (preferred)
custom_dataset/               # recorded CSI sessions (.npy)
examples/model_triangular_*.pkl   # legacy RSSI detector
```

### 2. Run the backend

```bash
pip install -r requirements.txt
cd examples
python rescue_backend.py
```

### 3. Run the dashboard

```bash
cd ui/react-rescue
npm install
npm run dev
```

Open `http://localhost:3000`. The dashboard connects to `ws://localhost:8002` (override with `VITE_WS_URL`).

### 4. Connect nodes (optional)

Flash the firmware (see `firmware/esp32-csi-node/README.md`), then provision each node:

```bash
python firmware/esp32-csi-node/provision.py --port COM7 \
  --ssid "<network>" --password "<password>" \
  --target-ip <laptop-ip> --target-port 4444 --node-id 1
```

Repeat with `--node-id 2` and `--node-id 3`. Use `python scripts/udp_debug.py` to confirm packets are arriving.

## Roadmap

1. **Reset and secure** — repository cleanup (this release).
2. **Data path** — shared wire-format spec, correct decoders, one UDP port, router-free anchor mode.
3. **Platform** — split the backend into tested stages with record/replay.
4. **Through-debris validation** — concrete-block test rig and measured detection rates.
5. **Algorithms, productisation, pilot** — gated on step 4.

## Research background

- **DensePose From WiFi** (CMU) — human pose estimation from WiFi CSI.
- **Widar 3.0** — cross-domain gesture recognition from WiFi.
- **SpotFi** (Stanford) — decimetre-level WiFi localisation.

## Team

Prins Kanyal · Vandana Bhandari · Aadarsh Dimri · Yuvraj Kabadwal

## License

MIT — see [LICENSE](LICENSE).
