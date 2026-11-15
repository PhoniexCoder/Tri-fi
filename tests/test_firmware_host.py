"""Firmware logic compiled and run on the host (no ESP-IDF needed)."""

import shutil
import subprocess

import pytest
from conftest import FIRMWARE_DIR

TEST_DIR = FIRMWARE_DIR / "test"
MAIN_DIR = FIRMWARE_DIR / "main"
CC = shutil.which("gcc") or shutil.which("clang")

pytestmark = pytest.mark.skipif(CC is None, reason="no host C compiler available")


def compile_c(sources, out, extra=()):
    cmd = [CC, "-std=gnu11", "-Wall", "-Werror", "-I", str(TEST_DIR / "stubs"),
           "-I", str(MAIN_DIR), *extra, *map(str, sources), "-o", str(out), "-lm"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_anchor_unit(tmp_path):
    exe = tmp_path / "anchor_test"
    compile_c([TEST_DIR / "anchor_test.c", MAIN_DIR / "anchor.c",
               TEST_DIR / "stubs" / "esp_stubs.c"], exe)
    proc = subprocess.run([str(exe)], capture_output=True, text=True)
    assert proc.returncode == 0 and "anchor_test: OK" in proc.stdout, proc.stdout


def test_nvs_config_compiles(tmp_path):
    proc = subprocess.run([CC, "-std=gnu11", "-Wall", "-Werror", "-fsyntax-only",
                           "-I", str(TEST_DIR / "stubs"), "-I", str(MAIN_DIR),
                           str(MAIN_DIR / "nvs_config.c")], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
