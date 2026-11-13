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
| Backend (Python) | `trifi/` | `python -m trifi`: UDP `4444` in, WebSocket `8002` out; record and replay sessions |
| Packet decoder | `trifi/protocol.py` | Mirrors the firmware wire structs exactly |
| Signal processing | `trifi/features.py` | Features and FFT breathing estimate |
| Kit config | `config/kit.example.toml` | Ports, API token, node positions, thresholds |
| Dashboard (React + Vite) | `ui/react-rescue/` | Dev server on port `3000` |
| Rescuer badge (ESP8266) | `firmware/rescuer_esp8266.ino` | Optional heartbeat / confirm button |
| Provisioning | `firmware/esp32-csi-node/provision.py` | Per-node WiFi, target, role (anchor/receiver) |
| Tools | `scripts/` | Rescuer flashing, UDP sniffer, mmWave bridge |
| Tests | `tests/` | `python -m pytest tests -q` (host C compiler needed for firmware tests) |

### Known limitations (being fixed)

- The backend's breathing filter suppresses the breathing band; one node is currently enough to report a detection.
- Positions shown are a weighted average of node locations, not true localisation.
- Node UDP traffic and OTA (without a PSK) are unauthenticated; the dashboard WebSocket supports a token.

## Quick start

### Requirements

- Python 3.11+ and Node.js 18+
- ESP-IDF v5.4 (only to build firmware)
- 3 or more ESP32-S3 boards (only for live data)

### 1. Get the models

Trained models and the recorded dataset are not stored in git. Download them from the project's GitHub Releases and unpack them into the repository root:

```
models/custom_presence_model.pth   # 2-class presence ResNet-18 (preferred)
models/model.pth                   # 6-class ResNet-18 (fallback)
custom_dataset/                    # recorded CSI (.npy)
```

Without a model the backend still runs, using the legacy RSSI detector only.

### 2. Run the backend

```bash
pip install -r requirements.txt
python -m trifi check                                # config + model preflight
python -m trifi run                                  # live
python -m trifi run --record                         # live, saving sessions/session-*.jsonl.gz
python -m trifi replay sessions/<file>.jsonl.gz      # replay a session to the dashboard
```

For a real kit, copy `config/kit.example.toml` to `config/kit.toml`, set an `api_token` and the node positions, and add `--config config/kit.toml` before the command.

### 3. Run the dashboard

```bash
cd ui/react-rescue
npm install
npm run dev
```

Open `http://localhost:3000` (with a token: `http://localhost:3000/?token=<api_token>`). The dashboard connects to port 8002 on the same host (override with `VITE_WS_URL`), so a tablet on the kit network can open `http://<laptop-ip>:3000` directly.

### 4. Connect nodes (optional)

Flash the firmware (see `firmware/esp32-csi-node/README.md`). All nodes and the laptop join one 2.4 GHz network; the laptop's own mobile hotspot works, so no router is needed.

Provision one node as the **anchor**. It broadcasts beacons at a fixed 100 Hz for the others to measure:

```bash
python firmware/esp32-csi-node/provision.py --port COM7 --role anchor \
  --ssid "<network>" --password "<password>" --node-id 0
```

Provision each **receiver**. It locks onto the anchor automatically and streams CSI to the laptop:

```bash
python firmware/esp32-csi-node/provision.py --port COM8 \
  --ssid "<network>" --password "<password>" \
  --target-ip <laptop-ip> --target-port 4444 --node-id 1
```

Repeat with `--node-id 2` and `--node-id 3`. On a Windows hotspot the laptop is usually `192.168.137.1`. Allow inbound UDP 4444 in the firewall. The backend logs each node's packet rate, loss and RSSI every 10 seconds; `python scripts/udp_debug.py` shows raw arrivals.

## Roadmap

1. **Reset and secure** — repository cleanup. Done.
2. **Data path** — correct decoders, one UDP port, fixed-rate anchor mode. Done, pending hardware verification.
3. **Platform** — backend split into tested stages, record/replay, dashboard v2, CI. Done.
4. **Through-debris validation** — concrete-block test rig and measured detection rates.
5. **Algorithms, productisation, pilot** — gated on step 4.

## Development

```bash
pip install -r requirements-dev.txt
ruff check trifi tests && mypy && python -m pytest
```

CI (`.github/workflows/ci.yml`) runs these, builds the dashboard, and builds the firmware for 8 MB and 4 MB boards with ESP-IDF v5.4.

## Research background

- **DensePose From WiFi** (CMU) — human pose estimation from WiFi CSI.
- **Widar 3.0** — cross-domain gesture recognition from WiFi.
- **SpotFi** (Stanford) — decimetre-level WiFi localisation.

## Team

Prins Kanyal · Vandana Bhandari · Aadarsh Dimri · Yuvraj Kabadwal

## License

MIT — see [LICENSE](LICENSE).
