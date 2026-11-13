"""Provisioning: NVS keys written must be the keys and types the firmware reads."""

import csv
import importlib.util
import io
import re

import pytest
from conftest import FIRMWARE_DIR

spec = importlib.util.spec_from_file_location("provision", FIRMWARE_DIR / "provision.py")
provision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provision)

NVS_TYPE = {"u8": "u8", "u16": "u16", "u32": "u32", "string": "str", "hex2bin": "blob"}


def rows(argv):
    parser = provision.build_parser()
    args = parser.parse_args(["--port", "COM1", *argv])
    provision.validate_args(parser, args)
    return {r["key"]: r for r in csv.DictReader(io.StringIO(provision.build_nvs_csv(args)))
            if r["type"] == "data"}


def firmware_nvs_keys():
    src = (FIRMWARE_DIR / "main" / "nvs_config.c").read_text()
    return dict((k, t) for t, k in re.findall(r'nvs_get_(u8|u16|u32|str|blob)\(handle, "(\w+)"', src))


def test_anchor_role_rows():
    r = rows(["--role", "anchor", "--anchor-hz", "120", "--node-id", "0"])
    assert r["node_role"]["value"] == "1"
    assert r["anchor_hz"]["value"] == "120"
    assert r["node_id"]["value"] == "0"


def test_receiver_lock_flags():
    assert rows(["--role", "receiver", "--no-anchor-lock"])["anchor_lock"]["value"] == "0"
    assert rows(["--anchor-lock"])["anchor_lock"]["value"] == "1"
    assert "anchor_lock" not in rows(["--node-id", "2"])


def test_wasm_options_merged():
    pk = "AB" * 32
    r = rows(["--no-wasm-verify", "--wasm-pubkey", pk])
    assert r["wasm_verify"]["value"] == "0"
    assert r["wasm_pubkey"]["value"] == pk.lower()


@pytest.mark.parametrize("argv", [
    ["--anchor-hz", "5"],
    ["--anchor-hz", "500"],
    ["--wasm-pubkey", "abc"],
    ["--role", "beacon"],
])
def test_invalid_arguments_rejected(argv):
    with pytest.raises(SystemExit):
        rows(argv)


def test_every_written_key_is_read_by_firmware_with_same_type():
    written = rows([
        "--ssid", "kit", "--password", "longpassword", "--target-ip", "192.168.137.1",
        "--target-port", "4444", "--node-id", "1", "--tdm-slot", "0", "--tdm-total", "3",
        "--edge-tier", "2", "--pres-thresh", "50", "--fall-thresh", "15000",
        "--vital-win", "256", "--vital-int", "1000", "--subk-count", "8", "--channel", "6",
        "--filter-mac", "24:6f:28:01:02:03", "--role", "receiver", "--anchor-hz", "100",
        "--anchor-lock", "--wasm-verify", "--wasm-pubkey", "00" * 32,
        "--seed-url", "http://x", "--seed-token", "t", "--zone", "z",
        "--swarm-hb", "30", "--swarm-ingest", "5",
    ])
    fw = firmware_nvs_keys()
    for key, row in written.items():
        assert key in fw, f"firmware never reads NVS key '{key}'"
        assert len(key) <= 15, f"NVS key '{key}' exceeds 15 characters"
        assert NVS_TYPE[row["encoding"]] == fw[key], f"type mismatch for '{key}'"
