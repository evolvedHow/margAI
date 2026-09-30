"""Config loading tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from margAI.config import DEFAULT_CONFIG_PATH, ConfigError, load_config

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
    path = tmp_path / "margAI.toml"
    path.write_text(text)
    return path


def test_loads_full_config(tmp_path):
    config = load_config(write(tmp_path), env={})
    assert config.gateway.prefix == "gm"
    assert config.gateway.expose == "both"
    assert config.gateway.timeout == 30
    assert config.telemetry.emit == "none"
    assert config.hooks.load == ("my.hooks",)
    names = {p.name for p in config.providers}
    assert {"main", "alt"} <= names
    main = provider(config, "main")
    assert main.api_key_env == "MY_KEY"
    assert main.models == ("a", "b")
    assert provider(config, "alt").default_model == "tiny"


def test_env_overrides_gateway(tmp_path):
    config = load_config(
        write(tmp_path),
        env={"MARGAI_PORT": "9090", "MARGAI_PREFIX": "mx", "MARGAI_HOST": "127.0.0.1"},
    )
    assert config.gateway.port == 9090
    assert config.gateway.prefix == "mx"
    assert config.gateway.host == "127.0.0.1"


def provider(config, name: str):
    """One provider by name.

    Looked up rather than indexed because the embedded default layer means
    `providers[0]` is no longer "the first provider in my file" -- and a test
    that silently depended on that ordering would break every time a default
    provider is added.
    """
    return next(p for p in config.providers if p.name == name)


def test_env_api_key_wins_over_env_name(tmp_path):
    config = load_config(write(tmp_path), env={"MY_KEY": "sk-real"})
    # api_key resolution happens at wrapper build; config keeps env name
    assert provider(config, "main").resolved_key({"MY_KEY": "sk-real"}) == "sk-real"
    assert provider(config, "main").resolved_key({}) is None


def test_absent_file_still_gets_the_embedded_defaults(tmp_path, monkeypatch):
    """A fresh install has to route something. The embedded layer is what makes
    that true, and this is the test that would notice if it stopped shipping."""
    monkeypatch.chdir(tmp_path)
    config = load_config(env={})
    assert config.gateway.prefix == "margAI"
    assert config.source is None
    assert config.layers == ("embedded",)
    assert {p.name for p in config.providers} >= {"openai"}


def test_invalid_expose_rejected(tmp_path):
    bad = TOML.replace('expose = "both"', 'expose = "sideways"')
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, bad), env={})


def test_invalid_env_expose_rejected(tmp_path):
    with pytest.raises(ConfigError) as e:
        load_config(write(tmp_path), env={"MARGAI_EXPOSE": "sideways"})
    assert "MARGAI_EXPOSE" in str(e.value)


def test_invalid_env_port_rejected(tmp_path):
    with pytest.raises(ConfigError) as e:
        load_config(write(tmp_path), env={"MARGAI_PORT": "not-a-number"})
    assert "MARGAI_PORT" in str(e.value)


def test_invalid_env_timeout_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_config(write(tmp_path), env={"MARGAI_TIMEOUT": "soon"})


def test_nonpositive_models_cache_ttl_rejected(tmp_path):
    bad = TOML.replace("timeout = 30", "timeout = 30\nmodels_cache_ttl = 0")
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, bad), env={})


def test_invalid_emit_rejected(tmp_path):
    bad = TOML.replace('emit = "none"', 'emit = "carrier-pigeon"')
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, bad), env={})


def test_file_emit_takes_dir_and_label(tmp_path):
    bad = TOML.replace('emit = "none"', 'emit = "file"\ndir = "~/logs"\nlabel = "vedanta"')
    config = load_config(write(tmp_path, bad), env={})
    assert config.telemetry.emit == "file"
    assert config.telemetry.dir == "~/logs"
    assert config.telemetry.label == "vedanta"


@pytest.mark.parametrize("bad_value", ["5", "true"])
def test_telemetry_dir_must_be_a_string(tmp_path, bad_value):
    bad = TOML.replace('emit = "none"', f'emit = "file"\ndir = {bad_value}')
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, bad), env={})


def test_provider_requires_kind_and_base_url(tmp_path):
    bad = """
    [providers.x]
    name = "no kind"
    """
    with pytest.raises(ConfigError):
        load_config(write(tmp_path, bad), env={})


def test_default_model_must_be_a_string_not_a_list(tmp_path):
    """A list here reaches Route(model=...) untouched and leaves as a JSON
    array in the upstream `model` field, so the provider 400s on something
    that is plainly a typo here."""
    bad = """
    [providers.x]
    kind = "openai"
    base_url = "https://api.test/v1"
    models = ["a", "b"]
    default_model = ["a"]
    """
    with pytest.raises(ConfigError) as e:
        load_config(write(tmp_path, bad), env={})
    assert "default_model" in str(e.value)
    assert "list" in str(e.value)


def test_a_valid_default_model_still_loads(tmp_path):
    good = """
    [providers.x]
    kind = "openai"
    base_url = "https://api.test/v1"
    models = ["a", "b"]
    default_model = "a"
    """
    assert provider(load_config(write(tmp_path, good), env={}), "x").default_model == "a"


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


PROVIDER_TOML = """
[providers.ollama]
kind = "openai-compatible"
base_url = "http://127.0.0.1:11434/v1"
"""


def test_provider_base_url_env_override(tmp_path):
    config = load_config(
        write(tmp_path, PROVIDER_TOML),
        env={"MARGAI_PROVIDER_BASE_URL_OLLAMA": "http://host.docker.internal:11435/v1"},
    )
    assert provider(config, "ollama").base_url == "http://host.docker.internal:11435/v1"


def test_provider_base_url_env_override_folds_non_alphanumerics(tmp_path):
    toml = PROVIDER_TOML.replace("providers.ollama", "providers.my-local-vllm")
    config = load_config(
        write(tmp_path, toml),
        env={"MARGAI_PROVIDER_BASE_URL_MY_LOCAL_VLLM": "http://10.0.0.5:8000/v1"},
    )
    assert provider(config, "my-local-vllm").base_url == "http://10.0.0.5:8000/v1"


def test_provider_base_url_env_override_ignored_when_blank(tmp_path):
    config = load_config(
        write(tmp_path, PROVIDER_TOML),
        env={"MARGAI_PROVIDER_BASE_URL_OLLAMA": "   "},
    )
    assert provider(config, "ollama").base_url == "http://127.0.0.1:11434/v1"


def test_provider_base_url_env_override_does_not_leak_across_providers(tmp_path):
    toml = """
    [providers.ollama]
    kind = "openai-compatible"
    base_url = "http://127.0.0.1:11434/v1"

    [providers.openai]
    kind = "openai"
    base_url = "https://api.openai.com/v1"
    """
    config = load_config(
        write(tmp_path, toml),
        env={"MARGAI_PROVIDER_BASE_URL_OLLAMA": "http://host.docker.internal:11435/v1"},
    )
    by_name = {p.name: p.base_url for p in config.providers}
    assert by_name["ollama"] == "http://host.docker.internal:11435/v1"
    assert by_name["openai"] == "https://api.openai.com/v1"


def test_provider_base_url_env_override_satisfies_missing_base_url(tmp_path):
    toml = """
    [providers.ollama]
    kind = "openai-compatible"
    """
    config = load_config(
        write(tmp_path, toml),
        env={"MARGAI_PROVIDER_BASE_URL_OLLAMA": "http://host.docker.internal:11435/v1"},
    )
    assert provider(config, "ollama").base_url == "http://host.docker.internal:11435/v1"


def test_margAI_config_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("MARGAI_CONFIG", str(write(tmp_path)))
    config = load_config(env=os.environ)
    assert config.source == str(tmp_path / "margAI.toml")

# --- layered configuration -------------------------------------------------
#
# The embedded _default.toml ships inside the package, so these assert the
# merge itself rather than any particular default: a test that only checked
# "openai is present" would keep passing after the layering silently broke.


def test_the_embedded_layer_is_shipped_inside_the_package():
    assert DEFAULT_CONFIG_PATH.is_file(), "embedded defaults are missing from the package"
    assert DEFAULT_CONFIG_PATH.name == "_default.toml"


def test_the_embedded_layer_parses_and_ships_no_hook_loads():
    """An example is not a default. Auto-loading code the operator never asked
    for, silently, on import, is the one thing defaults must not do."""
    import tomllib

    with DEFAULT_CONFIG_PATH.open("rb") as fh:
        raw = tomllib.load(fh)
    assert "hooks" not in raw
    assert "load" not in raw.get("hooks", {})


def test_the_embedded_layer_names_no_api_keys():
    import tomllib

    with DEFAULT_CONFIG_PATH.open("rb") as fh:
        raw = tomllib.load(fh)
    for name, table in raw.get("providers", {}).items():
        assert "api_key" not in table, f"{name} embeds a literal key"


def test_an_operators_file_overlays_the_embedded_layer(tmp_path):
    config = load_config(
        write(tmp_path, '[gateway]\ntimeout = 5.0\n'),
        env={},
    )
    assert config.gateway.timeout == 5.0
    # ...and the parts they said nothing about still arrive.
    assert config.gateway.prefix == "margAI"
    assert {p.name for p in config.providers} >= {"openai"}


def test_a_table_split_across_layers_arrives_whole(tmp_path):
    """Merged as raw TOML before parsing, so a provider that adds one key to a
    default provider is a complete provider, not a half-overridden one."""
    config = load_config(
        write(tmp_path, '[providers.openai]\nmodels = ["only-this"]\n'),
        env={},
    )
    openai = provider(config, "openai")
    assert openai.models == ("only-this",)
    assert openai.base_url == "https://api.openai.com/v1"
    assert openai.api_key_env == "OPENAI_API_KEY"


def test_lists_replace_rather_than_merge_across_layers(tmp_path):
    """Merging two model lists would invent a model neither layer declared."""
    config = load_config(
        write(tmp_path, '[providers.openai]\nmodels = ["a"]\n'),
        env={},
    )
    assert "gpt-4o" not in provider(config, "openai").models


def test_layers_records_provenance_for_the_doctor(tmp_path, monkeypatch):
    """The doctor has to be able to say "you did not configure this" as
    distinct from "there is no default for this"."""
    path = write(tmp_path)
    assert load_config(path, env={}).layers == ("embedded", str(path))
    # An empty directory: the cwd fallback must not find a file that is not there.
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    assert load_config(env={}).layers == ("embedded",)


def test_the_build_config_actually_ships_the_embedded_layer():
    """Guards the packaging, not the file.

    ``test_the_embedded_layer_is_shipped_inside_the_package`` reads the source
    tree, so it keeps passing while a broken wheel quietly omits the file --
    and an installed gateway with no embedded defaults starts up empty. The
    build failure this guards was real: listing `src/margAI/examples` as a
    package when it is already inside `src/margAI` made hatchling add one
    module twice and the build failed outright.
    """
    import tomllib

    root = Path(__file__).resolve().parent.parent
    with (root / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)

    listed = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert listed == ["src/margAI"], listed

    artifacts = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"].get("artifacts", [])
    assert "src/margAI/_default.toml" in artifacts, artifacts


# -- metering config: [telemetry] cycle + budgets ---------------------------


def test_reset_day_is_parsed_beside_the_prices(tmp_path):
    config = load_config(
        write(tmp_path, '[telemetry]\nreset_day = 15\n[telemetry.costs]\n"p/m" = { in = 1, out = 2 }\n'),
        env={},
    )
    assert config.billing.reset_day == 15
    assert config.telemetry.costs[0].model == "m"


def test_an_absent_reset_day_leaves_metering_off(tmp_path):
    config = load_config(write(tmp_path, TOML), env={})
    assert config.billing.reset_day is None
    assert config.billing.window() is None


@pytest.mark.parametrize("value", [0, 32, -1, 1.5, "1", True])
def test_a_bad_reset_day_is_rejected(tmp_path, value):
    literal = "true" if value is True else repr(value)
    with pytest.raises(ConfigError, match="reset_day"):
        load_config(write(tmp_path, f"[telemetry]\nreset_day = {literal}\n"), env={})


def test_period_bounds_must_be_given_together(tmp_path):
    with pytest.raises(ConfigError, match="both period_start and period_end"):
        load_config(write(tmp_path, '[telemetry]\nperiod_start = "2026-01-01"\n'), env={})


def test_period_start_must_not_be_after_the_end(tmp_path):
    with pytest.raises(ConfigError, match="after period_end"):
        load_config(
            write(tmp_path, '[telemetry]\nperiod_start = "2026-02-01"\nperiod_end = "2026-01-01"\n'),
            env={},
        )


def test_budgets_use_the_same_key_grammar_as_costs(tmp_path):
    config = load_config(
        write(
            tmp_path,
            "[telemetry]\nreset_day = 1\n"
            "[telemetry.budgets]\n"
            '"openai/gpt-4o" = 50.0\n'
            '"my_openai/*" = 100.0\n'
            '"gpt-4o-mini" = 20.0\n',
        ),
        env={},
    )
    by_key = {(e.provider, e.model): e.input_price for e in config.billing.budgets}
    assert by_key[("openai", "gpt-4o")] == 50.0
    assert by_key[("my_openai", "*")] == 100.0
    assert by_key[("*", "gpt-4o-mini")] == 20.0


def test_a_non_numeric_budget_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="must be a number"):
        load_config(write(tmp_path, '[telemetry.budgets]\n"p/m" = "lots"\n'), env={})


def test_an_unknown_telemetry_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(write(tmp_path, "[telemetry]\nreset_dai = 1\n"), env={})
