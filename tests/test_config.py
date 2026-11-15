"""Config loading and validation."""

import re

import pytest

from trifi.config import Config, ConfigError, describe, load_config


def write(tmp_path, text):
    p = tmp_path / "kit.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_defaults_are_valid():
    cfg = load_config()
    assert cfg.udp_port == 4444 and cfg.ws_port == 8002
    assert cfg.node_ids == (1, 2, 3)


def test_full_file(tmp_path):
    cfg = load_config(write(tmp_path, """
        [network]
        udp_port = 5555
        api_token = "secret"
        [nodes]
        1 = [0, 0]
        2 = [1, 0]
        3 = [0.5, 1]
        4 = [0.5, 0.5]
        [detection]
        op_mode = 2
        node_detect_threshold = 0.5
        [models]
        dir = "weights"
        [recording]
        enabled = true
        dir = "rec"
    """))
    assert cfg.udp_port == 5555 and cfg.api_token == "secret"
    assert cfg.node_ids == (1, 2, 3, 4)
    assert cfg.op_mode == 2 and cfg.node_detect_threshold == 0.5
    assert cfg.model_dir == (tmp_path / "weights").resolve()   # relative to the file
    assert cfg.record and cfg.session_dir == (tmp_path / "rec").resolve()


@pytest.mark.parametrize("text, message", [
    ("[network]\nudp_prot = 1", "unknown key"),
    ("[netwrk]\nudp_port = 1", "unknown config section"),
    ("[network]\nudp_port = 70000", "udp_port"),
    ("[network]\nudp_port = 'x'", "must be int"),
    ("[network]\nudp_port = 8002", "must differ"),
    ("[detection]\nop_mode = 7", "op_mode"),
    ("[detection]\nnode_detect_threshold = 1.5", "node_detect_threshold"),
    ("[nodes]\n0 = [0, 0]", "reserved for the anchor"),
    ("[nodes]\n1 = [2, 0]", "within [0, 1]"),
    ("[nodes]\n1 = [0]", "must be [x, y]"),
    ("not toml at all [", "cannot read"),
])
def test_invalid_files_rejected(tmp_path, text, message):
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(write(tmp_path, text))


def test_describe_redacts_token():
    d = describe(Config(api_token="hunter2"))
    assert d["api_token"] == "set"
    assert "hunter2" not in str(d)
