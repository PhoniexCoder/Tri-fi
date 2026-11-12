"""Shared pytest setup: make the backend modules importable."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIRMWARE_DIR = REPO_ROOT / "firmware" / "esp32-csi-node"

for p in (REPO_ROOT, REPO_ROOT / "examples"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
