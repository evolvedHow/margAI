"""Config loading tests."""

from __future__ import annotations

import os

import pytest

from margai.config import ConfigError, load_config

TOML = """
[gateway]
prefix = "gm"
expose = "both"
timeout = 30

[telemetry]
emit = "none"

[hooks]
load = ["my.hooks"]

[providers.main]
kind = "openai"
base_url = "https://api.example.com/v1"
api_key_env = "MY_KEY"
models = ["a", "b"]

[providers.alt]
kind = "openai-compatible"
base_url = "http://localhost:8080/v1"
default_model = "tiny"
"""


def write(tmp_path, text=TOML):
    path = tmp_path / "margai.toml"
    path.write_text(text)
    return path


def test_loads_full_config(tmp_path):
    config = load_config(write(tmp_path), env={})
    assert config.gateway.prefix == "gm"
    assert config.gateway.expose == "both"
    assert config.gateway.timeout == 30
    assert config.telemetry.emit == "none"
    assert config.hooks.load == ("my.hooks",)
    names = [p.name for p in config.providers]
    assert names == ["main", "alt"]
    main = config.providers[0]
    assert main.api_key_env == "MY_KEY"
    assert main.models == ("a", "b")
    assert config.providers[1].default_model == "tiny"


def test_env_overrides_gateway(tmp_path):
    config = load_config(
        write(tmp_path),
        env={"MARGAI_PORT": "9090", "MARGAI_PREFIX": "mx", "MARGAI_HOST": "127.0.0.1"},
    )
    assert config.gateway.port == 9090
    assert config.gateway.prefix == "mx"
    assert config.gateway.host == "127.0.0.1"


def test_env_api_key_wins_over_env_name(tmp_path):
    config = load_config(write(tmp_path), env={"MY_KEY": "sk-real"})
    # api_key resolution happens at wrapper build; config keeps env name
    assert config.providers[0].resolved_key({"MY_KEY": "sk-real"}) == "sk-real"
    assert config.providers[0].resolved_key({}) is None


def test_absent_file_uses_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config(env={})
    assert config.gateway.prefix == "marg"
    assert config.providers == ()


def test_invalid_expose_rejected(tmp_path):
    bad = TOML.replace('expose = "both"', 'expose = "sideways"')
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, bad), env={})


def test_invalid_emit_rejected(tmp_path):
    bad = TOML.replace('emit = "none"', 'emit = "carrier-pigeon"')
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, bad), env={})


def test_provider_requires_kind_and_base_url(tmp_path):
    bad = """
    [providers.x]
    name = "no kind"
    """
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, bad), env={})


def test_cost_entries(tmp_path):
    toml = """
    [telemetry]
    [telemetry.costs]
    "openai/gpt-4o" = { in = 2.5, out = 10.0 }
    "free-model" = { in = 0, out = 0 }
    """
    config = load_config(write(tmp_path, toml), env={})
    assert len(config.telemetry.costs) == 2
    entry = config.telemetry.costs[0]
    assert entry.provider == "openai"
    assert entry.model == "gpt-4o"
    assert entry.input_price == 2.5


def test_source_path_recorded(tmp_path):
    path = write(tmp_path)
    config = load_config(path, env={})
    assert config.source == str(path)


def test_margai_config_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("MARGAI_CONFIG", str(write(tmp_path)))
    config = load_config(env=os.environ)
    assert config.source == str(tmp_path / "margai.toml")