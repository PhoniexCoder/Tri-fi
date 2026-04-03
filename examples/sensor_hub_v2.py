"""
sensor_hub_v2.py — Full-Pipeline SensorHub with Raw CSI + Multi-Output ML

Replaces sensor_hub_ml.py (v1). Differences:
  1. Includes a UDP socket receiver for ADR-018 binary frames
  2. Integrates CSIFrameParser + CSIFeatureExtractor
  3. Uses PresenceDetectorV2 (presence + breathing depth + body zone)
  4. compute() returns 7 new fields: ml_prob, ml_detected,
     ml_breathing_depth, ml_body_zone, ml_zone_label, ml_zone_probs, ml_mode

Drop-in replacement in trifi_live.py:
    from sensor_hub_v2 import SensorHub, start_udp_csi_receiver

Usage in main():
    hub  = SensorHub()
    stop = threading.Event()
    # Start UDP receiver for raw CSI frames (in addition to serial)
    udp_thread = threading.Thread(
        target=start_udp_csi_receiver,
        args=(hub, "0.0.0.0", 4444, stop),
        daemon=True,
    )
    udp_thread.start()
"""

import collections
import socket
import threading
import time
import numpy as np

from csi_frame_parser     import CSIFrameParser, CSIFrame, VitalsPacket
from csi_feature_extractor import CSIFeatureExtractor
from ml_inference_rf       import PresenceDetectorRF

# ── Import original signal-processing helpers ─────────────────────────────────
try:
    from trifi_live import (
        WelfordStats, VitalAnomalyDetector, LongitudinalTracker,
        CoherenceScorer, HRVAnalyzer, BPEstimator, HappinessScorer, SeedBridge,
    )
except ImportError:
    class _Stub:
        def __init__(self, *a, **kw): pass
        def __getattr__(self, n):
            return lambda *a, **kw: {} if n in ("compute","summary") else None
    WelfordStats = VitalAnomalyDetector = LongitudinalTracker = _Stub
    CoherenceScorer = HRVAnalyzer = BPEstimator = HappinessScorer = SeedBridge = _Stub


# ─────────────────────────────────────────────────────────────────────────────
# UDP CSI receiver (runs in its own thread)
# ─────────────────────────────────────────────────────────────────────────────

def start_udp_csi_receiver(hub, bind_ip: str, bind_port: int, stop: threading.Event):
    """
    Listens for ADR-018 binary UDP frames from ESP32 nodes.
    Parses each frame and calls hub.update_csi_raw() with the decoded CSIFrame.

    Runs until stop.set() is called.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.settimeout(0.5)   # non-blocking poll so we can respect stop event
    sock.bind((bind_ip, bind_port))
    print(f"[UDP] Listening on {bind_ip}:{bind_port}")

    parser = CSIFrameParser()

    while not stop.is_set():
        try:
            data, addr = sock.recvfrom(4096)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"[UDP] recv error: {e}")
            continue

        result = parser.parse(data)

        if isinstance(result, CSIFrame):
            hub.update_csi_raw(result)
        elif isinstance(result, VitalsPacket):
            # Edge-computed vitals from edge_processing.c (0xC5110002)
            hub.update_csi(
                br       = result.breathing_bpm,
                motion   = result.motion_energy,
                presence = result.presence,
            )

    sock.close()
    print(f"[UDP] Receiver stopped. parsed={parser.n_parsed} dropped={parser.n_dropped}")


# ─────────────────────────────────────────────────────────────────────────────
# SensorHub
# ─────────────────────────────────────────────────────────────────────────────

class SensorHub:
    """
    Full-pipeline SensorHub with raw CSI → feature extraction → multi-output ML.

    Two parallel data paths:
      A) Serial / vitals path: update_csi(**kw) with derived scalars
         (backwards-compatible with original trifi_live.py reader)
      B) UDP binary path: update_csi_raw(CSIFrame) with raw I/Q frames
         → CSIFeatureExtractor → PresenceDetectorV2 (full resolution)

    Path B overrides Path A for presence/depth/zone when the model is ready.
    """

    def __init__(self, seed_url=None):
        self.lock = threading.Lock()

        # ── mmWave state ─────────────────────────────────────────────────────
        self.mw_hr       = 0.0
        self.mw_br       = 0.0
        self.mw_presence = False
        self.mw_distance = 0.0
        self.mw_lux      = 0.0
        self.mw_frames   = 0
        self.mw_ok       = False

        # ── CSI derived state (serial path) ──────────────────────────────────
        self.csi_hr       = 0.0
        self.csi_br       = 0.0
        self.csi_motion   = 0.0
        self.csi_presence = False
        self.csi_rssi     = 0
        self.csi_frames   = 0
        self.csi_ok       = False
        self.csi_fall     = False

        self.events = collections.deque(maxlen=50)

        # ── RuVector processors ───────────────────────────────────────────────
        self.hrv           = HRVAnalyzer()
        self.anomaly       = VitalAnomalyDetector()
        self.longitudinal  = LongitudinalTracker()
        self.coherence_mw  = CoherenceScorer()
        self.coherence_csi = CoherenceScorer()
        self.bp            = BPEstimator()
        self.happiness     = HappinessScorer()
        self.seed          = SeedBridge(seed_url) if seed_url else None
        self._last_seed_ingest = 0.0

        # ── Raw CSI pipeline (UDP path) ───────────────────────────────────────
        self._parser    = CSIFrameParser()
        self._extractor = CSIFeatureExtractor()
        self._detector  = PresenceDetectorRF(
            meta_path   = "model_rf_meta.json",
            threshold   = 0.60,
            smooth_n    = 5,
        )

        # Latest ML output snapshot (updated without holding hub lock)
        self._ml_status: dict = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Serial vitals path (unchanged API)
    # ──────────────────────────────────────────────────────────────────────────

    def update_mw(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, f"mw_{k}", v)
            self.mw_ok = True
            hr, br = kw.get("hr", 0), kw.get("br", 0)
            if hr > 0:
                self.hrv.add_hr(hr)
                self.longitudinal.observe("hr", hr)
                self.coherence_mw.update(1.0)
            else:
                self.coherence_mw.update(0.1)
            if br > 0:
                self.longitudinal.observe("br", br)
            alerts = self.anomaly.check(hr=hr, br=br)
            for a in alerts:
                self.events.append((time.time(), f"ANOMALY: {a[3]}"))

    def update_csi(self, **kw):
        """Serial-path update — also feeds v2 detector via update_simple()."""
        with self.lock:
            for k, v in kw.items():
                setattr(self, f"csi_{k}", v)
            self.csi_ok = True
            rssi      = kw.get("rssi",   self.csi_rssi)
            motion    = kw.get("motion", self.csi_motion)
            breathing = kw.get("br",     self.csi_br)
            if rssi != 0:
                self.longitudinal.observe("rssi", rssi)
                self.coherence_csi.update(min(1.0, max(0.0, (rssi + 90) / 50)))
            self.happiness.update(
                motion_energy = motion, br = breathing,
                hr = kw.get("hr", self.csi_hr), rssi = rssi,
            )
        # Infer from simple serial vitals if full raw frames aren't arriving
        # (The ML models support a simple 3-feature fallback)
        self._ml_status = self._detector.update_simple(rssi, motion, breathing)

    # ──────────────────────────────────────────────────────────────────────────
    # UDP raw I/Q path
    # ──────────────────────────────────────────────────────────────────────────

    def update_csi_raw(self, frame: CSIFrame):
        """
        Called by the UDP receiver with each decoded CSIFrame.
        Extracts a 27-dim feature vector and feeds it to the v2 detector.
        """
        # Read current vitals scalars (for Group A features)
        with self.lock:
            rssi      = frame.rssi
            motion    = self.csi_motion
            breathing = self.csi_br

        # Feature extraction (lock-free — extractor has its own state)
        feature_vec = self._extractor.update(frame, rssi, motion, breathing)

        if feature_vec is None:
            # PCA still warming up — fall through
            return

        # Inference (lock-free)
        self._ml_status = self._detector.update_from_feature_vector(feature_vec)

        # Update coherence from RSSI
        with self.lock:
            self.csi_rssi = rssi
            if rssi != 0:
                self.coherence_csi.update(min(1.0, max(0.0, (rssi + 90) / 50)))
                self.longitudinal.observe("rssi", float(rssi))

    def add_event(self, msg):
        with self.lock:
            self.events.append((time.time(), msg))

    # ──────────────────────────────────────────────────────────────────────────
    # Compute snapshot
    # ──────────────────────────────────────────────────────────────────────────

    def compute(self) -> dict:
        with self.lock:
            hrv   = self.hrv.compute()
            mw_hr = self.mw_hr
            csi_hr = self.csi_hr

            if mw_hr > 0 and csi_hr > 0:
                fused_hr, hr_src = mw_hr * 0.8 + csi_hr * 0.2, "Fused"
            elif mw_hr > 0:
                fused_hr, hr_src = mw_hr, "mmWave"
            elif csi_hr > 0:
                fused_hr, hr_src = csi_hr, "CSI"
            else:
                fused_hr, hr_src = 0, "—"

            mw_br, csi_br = self.mw_br, self.csi_br
            fused_br = (mw_br * 0.8 + csi_br * 0.2
                        if mw_br > 0 and csi_br > 0 else mw_br or csi_br)

            sbp, dbp = self.bp.estimate(fused_hr, hrv["sdnn"], hrv["lf_hf"])

            sdnn = hrv["sdnn"]
            stress = (
                "HIGH"     if 0 < sdnn < 30 else
                "Moderate" if sdnn < 50 else
                "Mild"     if sdnn < 80 else
                "Relaxed"  if sdnn < 100 else
                "Calm"     if sdnn >= 100 else "—"
            )

            drifts = []
            for metric, val in [("hr", fused_hr), ("br", fused_br),
                                 ("rssi", self.csi_rssi)]:
                if val:
                    d = self.longitudinal.check_drift(metric, val)
                    if d:
                        drifts.append(d)

            happy = self.happiness.compute()

            now = time.time()
            if self.seed and now - self._last_seed_ingest >= 5.0:
                self._last_seed_ingest = now
                self.seed.ingest(happy["vector"], {
                    "hr": fused_hr, "br": fused_br, "rssi": self.csi_rssi,
                    "presence": self.mw_presence or self.csi_presence,
                })

        # ── Final presence decision ───────────────────────────────────────────
        ml = self._ml_status
        ml_ready = ml.get("ml_window_filled", False)
        if ml_ready:
            final_presence = ml.get("ml_detected", False)
        else:
            final_presence = self.mw_presence or self.csi_presence

        return {
            # Original fields
            "hr": fused_hr,         "hr_src": hr_src,
            "br": fused_br,         "sbp": sbp,           "dbp": dbp,
            "stress": stress,       "sdnn": sdnn,          "rmssd": hrv["rmssd"],
            "pnn50": hrv["pnn50"],  "lf_hf": hrv["lf_hf"],
            "presence": final_presence,
            "distance": self.mw_distance,
            "lux": self.mw_lux,
            "rssi": self.csi_rssi,  "motion": self.csi_motion,
            "csi_frames": self.csi_frames,
            "mw_frames":  self.mw_frames,
            "coh_mw":  self.coherence_mw.score,
            "coh_csi": self.coherence_csi.score,
            "fall": self.csi_fall,
            "drifts": drifts,
            "events": list(self.events),
            "longitudinal": self.longitudinal.summary(),
            "happiness":        happy["happiness"],
            "gait_energy":      happy["gait_energy"],
            "affect_valence":   happy["affect_valence"],
            "social_energy":    happy["social_energy"],
            "happiness_vector": happy["vector"],

            # ── NEW v2 ML fields ──────────────────────────────────────────────
            "ml_prob":           ml.get("ml_prob",           0.0),
            "ml_detected":       ml.get("ml_detected",       False),
            "ml_breathing_depth":ml.get("ml_breathing_depth",0.0),
            "ml_body_zone":      ml.get("ml_body_zone",      0),
            "ml_zone_label":     ml.get("ml_zone_label",     "absent"),
            "ml_zone_probs":     ml.get("ml_zone_probs",     [1,0,0,0]),
            "ml_mode":           ml.get("ml_mode",           "rules"),
        }
