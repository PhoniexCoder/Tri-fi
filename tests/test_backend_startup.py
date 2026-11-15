"""Backend startup regressions (need torch and the model files)."""

import os
import subprocess
import sys

import pytest
from conftest import REPO_ROOT

MODEL_FILES = [REPO_ROOT / "models" / "custom_presence_model.pth", REPO_ROOT / "models" / "model.pth"]

pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not any(p.exists() for p in MODEL_FILES),
                                reason="model files not present (download from GitHub Releases)")


def test_check_loads_model_on_non_utf8_console():
    """A cp1252 console must not stop the model loading (it once did, silently)."""
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    proc = subprocess.run([sys.executable, "-m", "trifi", "check"], cwd=REPO_ROOT,
                          env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "classes) OK" in proc.stdout


def test_model_loads_with_weights_only():
    from trifi.inference import find_model, load_model

    model = load_model(find_model(MODEL_FILES))
    assert model.n_classes in (2, 6)
