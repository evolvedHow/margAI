"""Configuration loading for margAI.

The framework is configured through a single TOML file (``margAI.toml`` by
default) plus environment variables. The core keeps configuration pure
(``tomllib`` / dataclasses, stdlib only) so it stays portable.

File resolution order:
    1. The ``source`` argument passed to :func:`load_config`
    2. The ``MARGAI_CONFIG`` environment variable
    3. ``./margAI.toml`` in the working directory (if present)
    4. Built-in defaults (no file needed)

Gateway-level settings can also be overridden with ``MARGAI_*`` environment
variables (``MARGAI_HOST``, ``MARGAI_PORT``, ``MARGAI_PREFIX``,
``MARGAI_EXPOSE``, ``MARGAI_TIMEOUT``, ``MARGAI_DEFAULT_PROVIDER``,
``MARGAI_DYNAMIC_MODEL``).

A provider's upstream endpoint can be overridden per-provider with
``MARGAI_PROVIDER_BASE_URL_<NAME>`` (e.g. ``MARGAI_PROVIDER_BASE_URL_OLLAMA``).
This keeps deployment-specific endpoints out of the static TOML file.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Config",
    "ConfigError",
    "CostEntry",
    "GatewayConfig",
    "HookConfig",
    "ProviderConfig",
    "TelemetryConfig",
    "load_config",
]


class ConfigError(Exception):
    """Raised when the configuration is invalid or unusable."""


@dataclass(frozen=True)
class GatewayConfig:
    """Gateway-wide behavior."""

    prefix: str = "margAI"
    expose: str = "prefixed"
    timeout: float = 60.0
    host: str = "0.0.0.0"
    port: int = 8000
    default_provider: str | None = None
    models_cache_ttl: float = 300.0
    # The reserved model id whose target is picked per call by the selector
    # chain. Clients send "<prefix>/<this>" (e.g. "margAI/dynamic").
    dynamic_model: str = "dynamic"


@dataclass(frozen=True)
class ProviderConfig:
    """One upstream provider (OpenAI, OpenRouter, vLLM, Ollama, ...)."""

    name: str
    kind: str
    base_url: str
    api_key_env: str | None = None
    api_key: str | None = None
    default_model: str | None = None
    models: tuple[str, ...] = ()
    timeout: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def resolved_key(self, env: dict[str, str]) -> str | None:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return env.get(self.api_key_env)
        return None


@dataclass(frozen=True)
class CostEntry:
    provider: str
    model: str
    input_price: float
    output_price: float


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool = True
    emit: str = "log"
    callback: str | None = None
    costs: tuple[CostEntry, ...] = ()


@dataclass(frozen=True)
class HookConfig:
    load: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    providers: tuple[ProviderConfig, ...] = ()
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    hooks: HookConfig = field(default_factory=HookConfig)
    source: str | None = None


VALID_EXPOSE = {"prefixed", "both", "raw"}
VALID_EMIT = {"none", "log", "callback"}

_BOOL_ENV = {"1", "true", "yes", "on"}


def _env_overrides(env: dict[str, str]) -> dict[str, Any]:
    """Translate known ``MARGAI_*`` env vars into gateway fields."""
    out: dict[str, Any] = {}
    if (v := env.get("MARGAI_HOST")) is not None:
        out["host"] = v
    if (v := env.get("MARGAI_PORT")) is not None:
        try:
            out["port"] = int(v)
        except ValueError:
            raise ConfigError(f"MARGAI_PORT must be an integer, got '{v}'") from None
    if (v := env.get("MARGAI_PREFIX")) is not None:
        out["prefix"] = v
    if (v := env.get("MARGAI_EXPOSE")) is not None:
        if v not in VALID_EXPOSE:
            raise ConfigError(
                f"MARGAI_EXPOSE must be one of {sorted(VALID_EXPOSE)}, got '{v}'"
            )
        out["expose"] = v
    if (v := env.get("MARGAI_TIMEOUT")) is not None:
        try:
            out["timeout"] = float(v)
        except ValueError:
            raise ConfigError(f"MARGAI_TIMEOUT must be numeric, got '{v}'") from None
    if (v := env.get("MARGAI_DEFAULT_PROVIDER")) is not None:
        out["default_provider"] = v or None
    if (v := env.get("MARGAI_DYNAMIC_MODEL")) is not None:
        if not v.strip():
            raise ConfigError("MARGAI_DYNAMIC_MODEL must not be empty")
        out["dynamic_model"] = v.strip()
    return out


def _resolve_config_path(source: str | os.PathLike[str] | None, env: dict[str, str]) -> Path | None:
    if source is not None:
        return Path(source)
    if (path := env.get("MARGAI_CONFIG")) is not None:
        return Path(path)
    default = Path("margAI.toml")
    return default if default.exists() else None


def _base_url_override(name: str, env: dict[str, str]) -> str | None:
    """Per-provider ``base_url`` override from the environment.

    A provider's upstream lives in the TOML file, which is static and
    environment-agnostic. Deployment-specific endpoints (a local Ollama on
    another port, a LAN vLLM) therefore need an env escape hatch:

        MARGAI_PROVIDER_BASE_URL_OLLAMA=http://host.docker.internal:11435/v1

    The provider name is upper-cased with non-alphanumerics folded to ``_``.
    """
    key = "MARGAI_PROVIDER_BASE_URL_" + re.sub(r"[^A-Za-z0-9]+", "_", name).upper()
    value = env.get(key)
    return value.strip() if value and value.strip() else None


def _parse_providers(raw: dict[str, Any], env: dict[str, str]) -> list[ProviderConfig]:
    providers_tbl = raw.get("providers", {})
    if not isinstance(providers_tbl, dict):
        raise ConfigError("[providers] must be a table of providers")
    providers: list[ProviderConfig] = []
    for name, cfg in providers_tbl.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"provider '{name}' must be a table")
        kind = cfg.get("kind")
        base_url = _base_url_override(name, env) or cfg.get("base_url")
        if not kind or not isinstance(kind, str):
            raise ConfigError(f"provider '{name}': 'kind' is required")
        if not base_url or not isinstance(base_url, str):
            raise ConfigError(f"provider '{name}': 'base_url' is required")
        models = cfg.get("models") or []
        if isinstance(models, str):
            models = [models]
        if not all(isinstance(m, str) for m in models):
            raise ConfigError(f"provider '{name}': 'models' must be a list of strings")
        api_key_env = cfg.get("api_key_env")
        api_key = cfg.get("api_key")
        if api_key_env is not None and not isinstance(api_key_env, str):
            raise ConfigError(f"provider '{name}': 'api_key_env' must be a string")
        if api_key is not None and not isinstance(api_key, str):
            raise ConfigError(f"provider '{name}': 'api_key' must be a string")
        timeout = cfg.get("timeout")
        if timeout is not None and not isinstance(timeout, (int, float)):
            raise ConfigError(f"provider '{name}': 'timeout' must be numeric")
        if timeout is not None and float(timeout) <= 0:
            raise ConfigError(f"provider '{name}': 'timeout' must be positive")
        providers.append(
            ProviderConfig(
                name=name,
                kind=kind,
                base_url=base_url.rstrip("/"),
                api_key_env=api_key_env,
                api_key=api_key,
                default_model=cfg.get("default_model"),
                models=tuple(models),
                timeout=float(timeout) if timeout is not None else None,
                extra={k: v for k, v in cfg.items() if k not in {"kind", "base_url", "api_key_env", "api_key", "default_model", "models", "timeout"}},
            )
        )
    return providers


def _parse_costs(raw: dict[str, Any]) -> tuple[CostEntry, ...]:
    costs_tbl = raw.get("costs", {})
    if not isinstance(costs_tbl, dict):
        raise ConfigError("[telemetry.costs] must be a table")
    entries: list[CostEntry] = []
    for key, value in costs_tbl.items():
        if not isinstance(value, dict):
            raise ConfigError(f"[telemetry.costs] entry '{key}' must be a table")
        try:
            inp = float(value.get("in", 0.0) or 0.0)
            out = float(value.get("out", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"[telemetry.costs] entry '{key}' prices must be numeric") from exc
        if "/" in key:
            provider, _, model = key.partition("/")
        else:
            provider, model = "*", key
        entries.append(CostEntry(provider=provider, model=model, input_price=inp, output_price=out))
    return tuple(entries)


def load_config(
    source: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
) -> Config:
    """Load configuration from a TOML file (or defaults)."""
    env = dict(os.environ if env is None else env)
    path = _resolve_config_path(source, env)

    raw: dict[str, Any] = {}
    if path is not None:
        try:
            with path.open("rb") as fh:
                raw = tomllib.load(fh)
        except FileNotFoundError:
            pass
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"failed to parse config {path}: {exc}") from exc

    gateway_raw = dict(raw.get("gateway", {}))

    gateway_kwargs: dict[str, Any] = {
        "prefix": gateway_raw.pop("prefix", "margAI"),
        "expose": gateway_raw.pop("expose", "prefixed"),
        "timeout": gateway_raw.pop("timeout", 60.0),
        "host": gateway_raw.pop("host", "0.0.0.0"),
        "port": gateway_raw.pop("port", 8000),
        "default_provider": gateway_raw.pop("default_provider", None),
        "models_cache_ttl": gateway_raw.pop("models_cache_ttl", 300.0),
        "dynamic_model": gateway_raw.pop("dynamic_model", "dynamic"),
    }
    gateway_kwargs.update(_env_overrides(env))

    expose = str(gateway_kwargs["expose"])
    if expose not in VALID_EXPOSE:
        raise ConfigError(f"gateway.expose must be one of {sorted(VALID_EXPOSE)}, got '{expose}'")
    timeout = float(gateway_kwargs["timeout"])
    if timeout <= 0:
        raise ConfigError("gateway.timeout must be positive")
    cache_ttl = float(gateway_kwargs["models_cache_ttl"])
    if cache_ttl <= 0:
        raise ConfigError("gateway.models_cache_ttl must be positive")
    if not str(gateway_kwargs["dynamic_model"]).strip():
        raise ConfigError("gateway.dynamic_model must not be empty")
    gateway = GatewayConfig(**gateway_kwargs)

    telemetry_raw = dict(raw.get("telemetry", {}))
    emit = str(telemetry_raw.get("emit", "log"))
    if emit not in VALID_EMIT:
        raise ConfigError(f"telemetry.emit must be one of {sorted(VALID_EMIT)}, got '{emit}'")
    callback = telemetry_raw.get("callback")
    if callback is not None and not isinstance(callback, str):
        raise ConfigError("telemetry.callback must be a string")
    telemetry = TelemetryConfig(
        enabled=bool(telemetry_raw.get("enabled", True)),
        emit=emit,
        callback=callback,
        costs=_parse_costs(telemetry_raw),
    )

    hooks_raw = dict(raw.get("hooks", {}))
    load = hooks_raw.get("load") or ()
    if isinstance(load, str):
        load = [load]
    if not all(isinstance(item, str) for item in load):
        raise ConfigError("hooks.load must be a list of module strings")
    hooks = HookConfig(load=tuple(load))

    return Config(
        gateway=gateway,
        providers=tuple(_parse_providers(raw, env)),
        telemetry=telemetry,
        hooks=hooks,
        source=str(path) if path else None,
    )