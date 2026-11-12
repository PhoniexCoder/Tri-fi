"""Backend startup regressions."""

import os
import subprocess
import sys

import pytest

from conftest import REPO_ROOT

MODEL_FILES = [REPO_ROOT / "custom_presence_model.pth", REPO_ROOT / "model.pth"]


@pytest.mark.skipif(not any(p.exists() for p in MODEL_FILES),
                    reason="model files not present (download from GitHub Releases)")
def test_model_loads_on_non_utf8_console():
    """A cp1252 console must not stop the model loading (emoji in a log line)."""
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    code = "import rescue_backend as rb; print('MODEL_LOADED', rb.csi_node.model_loaded)"
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT / "examples",
                          env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    assert "MODEL_LOADED True" in proc.stdout, proc.stdout + proc.stderr
