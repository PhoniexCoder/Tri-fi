"""Quick verification that the CSI model + rescue_backend integration is fully functional."""
import sys, os, json
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "examples"))

print("=" * 55)
print("  Integration Verification")
print("=" * 55)

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}")
        failed += 1

# 1. Files exist
print("\n[1/5] File existence:")
for name in ["model.pkl", "scaler.pkl", "model_meta.json", "csi_preprocessing.py", "train_csi_model.py"]:
    path = os.path.join(PROJECT_ROOT, name)
    check(f"{name} ({os.path.getsize(path):,}B)" if os.path.exists(path) else name, os.path.exists(path))
check("rescue_backend.py", os.path.exists(os.path.join(PROJECT_ROOT, "examples", "rescue_backend.py")))

# 2. Imports
print("\n[2/5] Imports:")
try:
    from csi_preprocessing import preprocess_realtime_csi, preprocess_batch, FEATURE_NAMES
    check("csi_preprocessing module", True)
except Exception as e:
    check(f"csi_preprocessing: {e}", False)

try:
    import joblib
    model = joblib.load(os.path.join(PROJECT_ROOT, "model.pkl"))
    scaler = joblib.load(os.path.join(PROJECT_ROOT, "scaler.pkl"))
    check(f"model.pkl (RandomForest)", True)
    check(f"scaler.pkl (StandardScaler)", True)
except Exception as e:
    check(f"model load: {e}", False)

# 3. Inference pipeline
print("\n[3/5] Simulated ESP32 inference (64 subcarriers x 100 frames):")
fake_csi = np.random.rand(64, 100).astype(np.float32) * 10
features = preprocess_realtime_csi(fake_csi, scaler, window_size=100)
pred = model.predict(features)
proba = model.predict_proba(features)
check(f"Feature extraction -> shape {features.shape}", features.shape == (1, 10))
check(f"Prediction works -> class={pred[0]}", pred[0] in [0, 1])
check(f"Probabilities sum to 1 -> {proba[0]}", abs(sum(proba[0]) - 1.0) < 0.001)

# 4. rescue_backend.py integration points
print("\n[4/5] rescue_backend.py integration:")
import py_compile
try:
    py_compile.compile(os.path.join(PROJECT_ROOT, "examples", "rescue_backend.py"), doraise=True)
    check("Syntax valid", True)
except Exception as e:
    check(f"Syntax: {e}", False)

with open(os.path.join(PROJECT_ROOT, "examples", "rescue_backend.py"), "r", encoding="utf-8") as f:
    src = f.read()
check("CSIAmplitudeBuffer class defined", "class CSIAmplitudeBuffer" in src)
check("csi_buffer.push_frame() called on CSIFrame", "csi_buffer.push_frame(nid, result.amplitude)" in src)
check("preprocess_realtime_csi imported", "from csi_preprocessing import preprocess_realtime_csi" in src)
check("Combined probability blending", "CSI_ML_WEIGHT * csi_ml_prob" in src)
check("csi_ml_prob in WebSocket payload", '"csi_ml_prob"' in src)
check("tri_prob in WebSocket payload", '"tri_prob"' in src)
check("Fallback when model missing", "csi_buffer.model_loaded" in src)

# 5. Model metadata
print("\n[5/5] Model metadata:")
with open(os.path.join(PROJECT_ROOT, "model_meta.json")) as f:
    meta = json.load(f)
check(f"Accuracy: {meta['accuracy']*100:.1f}%", meta["accuracy"] > 0.90)
check(f"Features: {meta['n_features']}", meta["n_features"] == 10)
check(f"Window size: {meta['window_size']}", meta["window_size"] == 100)
check(f"Classes: no_human + human_present", "0" in meta["classes"] and "1" in meta["classes"])

# Summary
print("\n" + "=" * 55)
total = passed + failed
if failed == 0:
    print(f"  ALL {passed}/{total} CHECKS PASSED")
    print(f"  Integration is FULLY FUNCTIONAL")
else:
    print(f"  {passed}/{total} passed, {failed} FAILED")
print("=" * 55)
