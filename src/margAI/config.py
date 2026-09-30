"""Configuration loading for margAI.

The framework is configured through a single TOML file (``margAI.toml`` by
default) plus environment variables. The core keeps configuration pure
(``tomllib`` / dataclasses, stdlib only) so it stays portable.

File resolution order:
    1. The ``source`` argument passed to :func:`load_config`
    2. The ``MARGAI_CONFIG`` environment variable
    3. ``./margAI.toml`` in the working directory (if present)

Each of those is an *overlay* on top of ``_default.toml``, which ships inside
the package. So the effective precedence, highest first, is:

    per-call config  >  margAI.toml / MARGAI_CONFIG  >  MARGAI_* env  >
    embedded _default.toml  >  dataclass defaults

The embedded layer is what makes a fresh install work without a config file.
It is raw-TOML merged rather than a set of dataclass defaults so that a new
``[providers.*]`` or ``[models.*]`` entry ships as a table an operator can
override one key of, instead of a new field that only the dataclasses know
about. See :func:`default_config_raw`.

Gateway-level settings can also be overridden with ``MARGAI_*`` environment
variables (``MARGAI_HOST``, ``MARGAI_PORT``, ``MARGAI_PREFIX``,
``MARGAI_EXPOSE``, ``MARGAI_TIMEOUT``, ``MARGAI_DEFAULT_PROVIDER``,
``MARGAI_DYNAMIC_MODEL``).

A provider's upstream endpoint can be overridden per-provider with
``MARGAI_PROVIDER_BASE_URL_<NAME>`` (e.g. ``MARGAI_PROVIDER_BASE_URL_OLLAMA``).
This keeps deployment-specific endpoints out of the static TOML file.
"""

from __future__ import annotations

import copy
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "BillingConfig",
    "Config",
    "ConfigError",
    "CostEntry",
    "GatewayConfig",
    "HookConfig",
    "ModelPolicy",
    "ProviderConfig",
    "TelemetryConfig",
    "default_config_raw",
    "load_config",
]

#: Selection strategies a virtual model may declare. ``first`` is deterministic
#: and therefore the default; ``cheapest`` needs prices to rank by;
#: ``round_robin`` exists for spreading load, so it is the one strategy that is
#: *not* a pure function of its arguments; ``highest_balance`` spends what
#: margAI has estimated against a declared budget; and ``least_used`` spreads
#: load by call count and needs no configuration at all.
VALID_STRATEGIES = {
    "first",
    "cheapest",
    "round_robin",
    "highest_balance",
    "least_used",
}


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
    # What to do with a bangtag no registered handler claims. The framework
    # cannot tell "typo" from "a pack that is not installed here", which is
    # why this is a policy and not a rule -- see Wrapper._check_tags.
    on_unknown_tag: str = "warn"
    # Scan the `margAI.packs` entry point group for installed packs. On by
    # default: a pack that has to be imported by hand is a pack that silently
    # does nothing when someone forgets the import.
    pack_discovery: bool = True
    # Install only these packs. None means "every discovered pack"; an
    # explicit list is a request, so a named pack that fails to install is an
    # error rather than a log line.
    packs_only: tuple[str, ...] | None = None


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
    #: ``none`` | ``log`` | ``callback`` | ``file``.
    emit: str = "log"
    callback: str | None = None
    costs: tuple[CostEntry, ...] = ()
    #: Directory for ``emit = "file"``. ``None`` means the standard per-user
    #: location (``$XDG_STATE_HOME/margAI/logs``, else ``~/.local/state/...``).
    dir: str | None = None
    #: Base name of the telemetry log, e.g. ``vedanta`` in
    #: ``vedanta_telemetry_260929143022.log``. ``None`` falls back to the
    #: gateway prefix, so the file names the app even when unset.
    label: str | None = None


@dataclass(frozen=True)
class BillingConfig:
    """Spend caps and the month they are measured over.

    margAI cannot read a provider's real account balance -- there is no such
    API it can call generically, and the core deliberately does no I/O. So
    "how much is left" is defined here as *declared budget minus the spend this
    gateway has estimated*, which is the only figure it can actually know.

    That makes the numbers an estimate rather than an invoice: they come from
    the same ``[telemetry.costs]`` table cost reports use, and a model with no
    price contributes nothing. ``highest_balance`` is therefore a way of
    spreading load away from a budget you are about to hit, not a way of
    querying one.

    Billing is assumed monthly and recurring, so the one setting that matters
    is ``reset_day``: the day of the month the cycle starts. That rolls
    forward on its own, so a gateway left running does not need editing when
    the month turns over -- and unlike a pinned month it cannot go stale and
    leave every candidate ranking as fully unspent.

    It has **no default**, and that is deliberate. ``highest_balance`` is only
    meaningful when spend can actually be *metered*, which needs two things
    that both have to be present: prices in ``[telemetry.costs]``, and a
    ``reset_day`` here. Either one alone is not enough -- prices without a
    cycle cannot say what has been spent *this month*, and a cycle without
    prices divides by an unknown. Absent either, the strategy falls back to
    ``least_used`` rather than inventing a number, so an operator who never
    configured billing gets sane spreading instead of silent nonsense.

    A ``reset_day`` past the end of a short month is clamped rather than
    rejected: ``31`` in February means the 28th (or 29th), so the cycle is
    always well-formed. Refusing it instead would make the obvious value --
    "my provider bills on the 31st" -- unconfigurable.

    ``period_start`` / ``period_end`` are the escape hatch for a provider that
    does not bill monthly at all. Both must be given together, and they
    override ``reset_day`` rather than adding to it.
    """

    #: Day of month the cycle starts, 1-31. Clamped to the month's last day.
    #: ``None`` means "no billing cycle configured", which disables metering.
    reset_day: int | None = None
    #: ISO ``YYYY-MM-DD``; spend is counted from this instant.
    period_start: str | None = None
    #: ISO ``YYYY-MM-DD``; inclusive, so the budget covers the whole day.
    period_end: str | None = None
    #: ``$`` caps keyed ``provider/model`` or bare ``model``, same shape and
    #: same "most specific wins" rule as ``[telemetry.costs]``.
    budgets: tuple[CostEntry, ...] = ()

    def window(self, now: float | None = None) -> tuple[float, float] | None:
        """The ``[start, end)`` epoch seconds budgets are spent against.

        ``None`` when no cycle is configured at all -- no explicit period and
        no ``reset_day``. Callers read ``None`` as "metering is off", which is
        the fallback signal to ``least_used``.
        """
        if self.period_start and self.period_end:
            start = date.fromisoformat(self.period_start)
            end = date.fromisoformat(self.period_end)
            return (
                datetime.combine(start, time.min, tzinfo=UTC).timestamp(),
                # Inclusive end date: the budget covers the whole of that day.
                datetime.combine(end, time.max, tzinfo=UTC).timestamp(),
            )
        if self.reset_day is None:
            return None
        current = datetime.now(UTC).timestamp() if now is None else now
        today = datetime.fromtimestamp(current, UTC).date()
        # The most recent occurrence of the reset day, on or before today.
        anchor = _month_day(today.year, today.month, self.reset_day)
        if anchor > today:
            anchor = _month_day(today.year - (1 if today.month == 1 else 0),
                                12 if today.month == 1 else today.month - 1,
                                self.reset_day)
        following = _add_months(anchor, 1)
        return (
            datetime.combine(anchor, time.min, tzinfo=UTC).timestamp(),
            datetime.combine(following, time.min, tzinfo=UTC).timestamp(),
        )

    def is_current(self, now: float | None = None) -> bool:
        """Whether ``now`` falls inside the configured billing window.

        False for an explicit period that has already passed: that is a stale
        config, not a zero balance, and ranking against it would claim every
        candidate is unspent forever. Callers treat ``False`` as "fall back to
        ``least_used``", so the stale case degrades instead of lying.
        """
        window = self.window(now)
        if window is None:
            return False
        current = datetime.now(UTC).timestamp() if now is None else now
        return window[0] <= current <= window[1]


@dataclass(frozen=True)
class ModelPolicy:
    """One virtual model: a name clients send instead of a real model id.

    ``margAI/fast`` is not a model anyone hosts. It is a *claim* about one --
    "cheap, on these providers, preferring that one" -- and the gateway
    resolves it per call against the models actually configured. That keeps the
    client contract stable while the model catalogue underneath it changes, and
    it means a caller's model id is a preference rather than a promise.

    The fields are filters (which candidates are eligible) plus a strategy
    (how one is picked from those left). The call's own ``not=``/``only=``/
    ``route=`` constraints are applied *after* these, so a policy can widen the
    field but never widen away a caller's exclusion.
    """

    name: str
    #: How to pick from the eligible candidates.
    strategy: str = "first"
    #: Eligible only if the call carries every one of these tags. Empty means
    #: always eligible, which is what a plain alias wants.
    tags: tuple[str, ...] = ()
    #: Provider allow-list. Empty means any.
    providers: tuple[str, ...] = ()
    #: Provider deny-list, applied after the allow-list.
    exclude_providers: tuple[str, ...] = ()
    #: Model allow-list, as bare model ids or ``provider/model`` pairs.
    models: tuple[str, ...] = ()
    #: Tried in order before the strategy runs. An id that matches nothing is
    #: skipped rather than fatal, so a preference can outlive the model it named.
    prefer: tuple[str, ...] = ()
    #: Ceiling on blended $ per 1M tokens. Requires prices in
    #: ``[telemetry.costs]``; a candidate with no known price is excluded when
    #: a ceiling is set, because "unknown" cannot be shown to be under it.
    max_cost_per_1m: float | None = None

    def id(self, prefix: str) -> str:
        return f"{prefix}/{self.name}"


@dataclass(frozen=True)
class HookConfig:
    load: tuple[str, ...] = ()


@dataclass(frozen=True)
class Config:
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    providers: tuple[ProviderConfig, ...] = ()
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    billing: BillingConfig = field(default_factory=BillingConfig)
    hooks: HookConfig = field(default_factory=HookConfig)
    #: Virtual models, keyed by bare name. Order is the declaration order,
    #: which is the tie-break when two policies both match a call.
    models: Mapping[str, ModelPolicy] = field(default_factory=dict)
    #: Per-pack configuration, keyed by pack name. This is the *base* a
    #: caller's per-call config merges onto; see core.configview.
    packs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    source: str | None = None
    #: Which layers contributed, lowest first, for `margAI doctor`. The
    #: embedded layer is always present, so an operator can always tell the
    #: difference between "I did not configure this" and "there is no
    #: default for this".
    layers: tuple[str, ...] = ()


VALID_EXPOSE = {"prefixed", "both", "raw"}
VALID_EMIT = {"none", "log", "callback", "file"}
#: What to do with a bangtag no handler claims. `ignore` is the escape hatch
#: for a gateway that runs packs whose tags are not all installed locally.
VALID_UNKNOWN_TAG = {"warn", "error", "ignore"}

# Provider table keys that are lifted onto ProviderConfig fields; everything
# else falls through to `extra`.
_KNOWN_PROVIDER_KEYS = {
    "kind",
    "base_url",
    "api_key_env",
    "api_key",
    "default_model",
    "models",
    "timeout",
}

_BOOL_ENV = {"1", "true", "yes", "on"}

MAX_MERGE_DEPTH = 12


#: The embedded default config, parsed once. Small and import-time-cached; a
#: fresh import is not a thing to pay for on the request path.
_EMBEDDED_DEFAULTS: dict[str, Any] | None = None

DEFAULT_CONFIG_PATH = Path(__file__).with_name("_default.toml")


def default_config_raw() -> dict[str, Any]:
    """The embedded ``_default.toml``, as a raw table.

    A missing or malformed embedded file is a packaging bug, not a deployment
    problem, so it raises rather than degrading to "no defaults" -- a gateway
    that silently came up with no providers would fail every request with an
    error that points nowhere near the real cause.
    """
    global _EMBEDDED_DEFAULTS
    if _EMBEDDED_DEFAULTS is None:
        try:
            with DEFAULT_CONFIG_PATH.open("rb") as fh:
                _EMBEDDED_DEFAULTS = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(
                f"cannot read the embedded default config at {DEFAULT_CONFIG_PATH}: {exc}"
            ) from exc
    # A fresh copy per call: the merge in load_config must not be able to reach
    # back and mutate the cached table.
    return copy.deepcopy(_EMBEDDED_DEFAULTS)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any], _depth: int = 0) -> dict[str, Any]:
    """Merge ``overlay`` onto ``base``: tables recurse, everything else replaces.

    The same asymmetry as the per-call overlay, and for the same reason --
    merging two lists of providers would invent a provider neither file
    declared. The depth bound is for a recursive structure handed in by
    something other than a TOML parser, not for TOML itself.
    """
    if _depth > MAX_MERGE_DEPTH:
        raise ConfigError("configuration nests too deeply to merge")
    out = dict(base)
    for key, value in overlay.items():
        current = out.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            out[key] = _deep_merge(current, value, _depth + 1)
        else:
            out[key] = value
    return out


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


def _resolve_config_path(
    source: str | os.PathLike[str] | None, env: dict[str, str]
) -> tuple[Path | None, bool]:
    """The config file to read, and whether it was asked for.

    The flag matters because a missing file has two meanings. ``./margAI.toml``
    being absent is the normal no-config case; a path the operator *named* --
    ``-c``, or ``MARGAI_CONFIG`` -- being absent is a typo, and silently
    falling through to the embedded defaults would start a gateway with none
    of their providers and no indication why.
    """
    if source is not None:
        return Path(source), True
    if (path := env.get("MARGAI_CONFIG")) is not None:
        return Path(path), True
    default = Path("margAI.toml")
    return (default, False) if default.exists() else (None, False)


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
        # Every other provider field is type-checked, and this one matters as
        # much: a list here flows straight into Route(model=...) and out as a
        # JSON array in the upstream `model` field, which fails as a confusing
        # 400 from the provider rather than as a config error.
        default_model = cfg.get("default_model")
        if default_model is not None and not isinstance(default_model, str):
            raise ConfigError(f"provider '{name}': 'default_model' must be a single string, not a list")
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
                default_model=default_model,
                models=tuple(models),
                timeout=float(timeout) if timeout is not None else None,
                extra={
                    k: v
                    for k, v in cfg.items()
                    if k not in _KNOWN_PROVIDER_KEYS
                },
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


def _month_day(year: int, month: int, day: int) -> date:
    """Day ``day`` of ``year``-``month``, clamped to that month's last day.

    A reset day of 31 has to mean something in February, and "the last day of
    the month" is the only reading an operator who bills on the 31st would
    accept. Rejecting it would make the natural value unconfigurable.
    """
    last = _days_in_month(year, month)
    return date(year, month, min(day, last))


def _days_in_month(year: int, month: int) -> int:
    return (date(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1) - timedelta(days=1)).day


def _add_months(value: date, months: int) -> date:
    """``value`` shifted by whole months, clamped so the day stays valid."""
    index = value.year * 12 + (value.month - 1) + months
    return _month_day(index // 12, index % 12 + 1, value.day)


def _iso_date(field: str, value: Any) -> date:
    """Parse a ``[telemetry]`` date field, naming it in any error."""
    if not isinstance(value, str):
        raise ConfigError(f"telemetry.{field} must be an ISO 'YYYY-MM-DD' string")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise ConfigError(
            f"telemetry.{field} must be an ISO 'YYYY-MM-DD' date, got '{value}'"
        ) from None


def _parse_billing(telemetry_raw: dict[str, Any]) -> BillingConfig:
    """Parse the metering half of ``[telemetry]``: cycle plus ``$`` caps.

    Metering needs two things that live together here. Prices (``costs``) say
    what a token costs; ``reset_day`` says which cycle the spend is counted
    against. *Both* must be present for ``highest_balance`` to mean anything,
    which is why the reset day sits beside the prices rather than in a section
    of its own -- an operator configuring one is configuring the other.

    ``reset_day`` is a plain day of the month, not a pinned calendar month: it
    rolls forward by itself and cannot go stale. The explicit
    ``period_start`` / ``period_end`` pair is for a provider that does not bill
    monthly, and both are required together -- a period with only one bound has
    no meaning, and treating it as open-ended would let a stale budget keep
    steering routing indefinitely.
    """
    reset_day = telemetry_raw.get("reset_day")
    if reset_day is not None:
        if isinstance(reset_day, bool) or not isinstance(reset_day, int):
            raise ConfigError(
                f"telemetry.reset_day must be an integer day of the month, got {reset_day!r}"
            )
        if not 1 <= reset_day <= 31:
            raise ConfigError(f"telemetry.reset_day must be 1-31, got {reset_day}")

    start = telemetry_raw.get("period_start")
    end = telemetry_raw.get("period_end")
    if (start is None) != (end is None):
        raise ConfigError(
            "[telemetry] needs both period_start and period_end, or neither; "
            f"got period_start={start!r}, period_end={end!r}"
        )
    if start is not None:
        start_day = _iso_date("period_start", start)
        end_day = _iso_date("period_end", end)
        if start_day > end_day:
            raise ConfigError(
                f"telemetry.period_start ({start}) is after period_end ({end})"
            )

    budgets_tbl = telemetry_raw.get("budgets", {})
    if not isinstance(budgets_tbl, dict):
        raise ConfigError("[telemetry.budgets] must be a table")
    budgets: list[CostEntry] = []
    for key, value in budgets_tbl.items():
        # A bare number is the common case: `budgets."openai/gpt-4o" = 50.0`.
        amount = value.get("usd") if isinstance(value, dict) else value
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise ConfigError(
                f"[telemetry.budgets] entry '{key}' must be a number of $ "
                "(or a table with a 'usd' key)"
            )
        if float(amount) < 0:
            raise ConfigError(f"[telemetry.budgets] entry '{key}' must not be negative")
        # Same key grammar as [telemetry.costs]: `provider/model`, a bare
        # `model` for any provider, or `provider/*` for a whole provider.
        if "/" in key:
            provider, _, model = key.partition("/")
        else:
            provider, model = "*", key
        budgets.append(
            CostEntry(
                provider=provider,
                model=model,
                input_price=float(amount),
                output_price=0.0,
            )
        )
    return BillingConfig(
        reset_day=reset_day,
        period_start=start,
        period_end=end,
        budgets=tuple(budgets),
    )


def _str_tuple(value: Any, where: str) -> tuple[str, ...]:
    """Coerce a scalar or list of scalars into a tuple of strings."""
    if value is None:
        return ()
    items = [value] if isinstance(value, str) else value
    if not isinstance(items, (list, tuple)):
        raise ConfigError(f"{where} must be a string or a list of strings")
    if not all(isinstance(item, str) for item in items):
        raise ConfigError(f"{where} must be a string or a list of strings")
    return tuple(str(item).strip() for item in items if str(item).strip())


def _parse_packs(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Parse ``[packs.<name>]`` tables into plain data.

    Held as data rather than as a validated schema on purpose: a pack's
    settings are the pack's business, and a framework that enumerated them
    would need a release every time a pack added an option. The gateway only
    enforces the *shape* -- a table of tables -- because that is what the
    per-call overlay has to be able to merge into.
    """
    tbl = raw.get("packs", {})
    if not isinstance(tbl, dict):
        raise ConfigError("[packs] must be a table of packs")
    out: dict[str, dict[str, Any]] = {}
    for name, cfg in tbl.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"[packs.{name}] must be a table")
        out[name] = dict(cfg)
    return out


def _parse_models(raw: dict[str, Any]) -> dict[str, ModelPolicy]:
    """Parse ``[models.<name>]`` virtual-model tables.

    A malformed policy is a load error rather than a warning: it is a claim
    about routing, and silently dropping it would send a caller's request to a
    model they did not ask for.
    """
    tbl = raw.get("models", {})
    if not isinstance(tbl, dict):
        raise ConfigError("[models] must be a table of virtual models")
    out: dict[str, ModelPolicy] = {}
    for name, cfg in tbl.items():
        if not isinstance(cfg, dict):
            raise ConfigError(f"[models.{name}] must be a table")
        strategy = str(cfg.get("strategy", "first"))
        if strategy not in VALID_STRATEGIES:
            raise ConfigError(
                f"[models.{name}] strategy must be one of {sorted(VALID_STRATEGIES)}, "
                f"got '{strategy}'"
            )
        max_cost = cfg.get("max_cost_per_1m")
        if max_cost is not None:
            if not isinstance(max_cost, (int, float)) or isinstance(max_cost, bool):
                raise ConfigError(f"[models.{name}] max_cost_per_1m must be numeric")
            if float(max_cost) < 0:
                raise ConfigError(f"[models.{name}] max_cost_per_1m must not be negative")
            max_cost = float(max_cost)
        unknown = set(cfg) - {
            "strategy",
            "tags",
            "providers",
            "exclude_providers",
            "models",
            "prefer",
            "max_cost_per_1m",
        }
        if unknown:
            raise ConfigError(
                f"[models.{name}] has unknown key(s) {sorted(unknown)}; "
                "a virtual model is a routing claim, not a place for arbitrary data"
            )
        out[name] = ModelPolicy(
            name=name,
            strategy=strategy,
            tags=_str_tuple(cfg.get("tags"), f"[models.{name}] tags"),
            providers=_str_tuple(cfg.get("providers"), f"[models.{name}] providers"),
            exclude_providers=_str_tuple(
                cfg.get("exclude_providers"), f"[models.{name}] exclude_providers"
            ),
            models=_str_tuple(cfg.get("models"), f"[models.{name}] models"),
            prefer=_str_tuple(cfg.get("prefer"), f"[models.{name}] prefer"),
            max_cost_per_1m=max_cost,
        )
    return out


def load_config(
    source: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> Config:
    """Load configuration from a TOML file (or defaults)."""
    env = dict(os.environ if env is None else env)
    path, explicit = _resolve_config_path(source, env)

    file_raw: dict[str, Any] = {}
    if path is not None:
        try:
            with path.open("rb") as fh:
                file_raw = tomllib.load(fh)
        except FileNotFoundError:
            if explicit:
                raise ConfigError(f"config file not found: {path}") from None
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"failed to parse config {path}: {exc}") from exc

    # The embedded layer sits underneath the operator's file, so the file only
    # has to state what it changes. Parsed *after* the merge, so a provider
    # table split across both layers arrives whole.
    raw = _deep_merge(default_config_raw(), file_raw)

    gateway_raw = dict(raw.get("gateway", {}))

    gateway_kwargs: dict[str, Any] = {
        "prefix": gateway_raw.pop("prefix", "margAI"),
        "expose": gateway_raw.pop("expose", "prefixed"),
        "timeout": gateway_raw.pop("timeout", 60.0),
        "host": gateway_raw.pop("host", "0.0.0.0"),
        "port": gateway_raw.pop("port", 8000),
        # Empty string means "unset", matching the dataclass default and the
        # env override. Left as `""` it would be a truthy-looking value that
        # names no provider.
        "default_provider": gateway_raw.pop("default_provider", None) or None,
        "models_cache_ttl": gateway_raw.pop("models_cache_ttl", 300.0),
        "dynamic_model": gateway_raw.pop("dynamic_model", "dynamic"),
        "on_unknown_tag": gateway_raw.pop("on_unknown_tag", "warn"),
        "pack_discovery": gateway_raw.pop("pack_discovery", True),
        "packs_only": _str_tuple(gateway_raw.pop("packs_only", None), "gateway.packs_only") or None,
    }
    gateway_kwargs.update(_env_overrides(env))

    port = gateway_kwargs["port"]
    if not isinstance(port, int) or isinstance(port, bool) or not (0 < port < 65536):
        raise ConfigError(f"gateway.port must be an integer in 1..65535, got {port!r}")
    host = gateway_kwargs["host"]
    if not isinstance(host, str) or not host.strip():
        raise ConfigError("gateway.host must be a non-empty string")
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
    unknown_tag = str(gateway_kwargs["on_unknown_tag"])
    if unknown_tag not in VALID_UNKNOWN_TAG:
        raise ConfigError(
            f"gateway.on_unknown_tag must be one of {sorted(VALID_UNKNOWN_TAG)}, "
            f"got '{unknown_tag}'"
        )
    gateway = GatewayConfig(**gateway_kwargs)

    telemetry_raw = dict(raw.get("telemetry", {}))
    unknown_telemetry = set(telemetry_raw) - {
        "enabled", "emit", "callback", "costs", "dir", "label",
        "reset_day", "period_start", "period_end", "budgets",
    }
    if unknown_telemetry:
        raise ConfigError(
            f"[telemetry] has unknown key(s) {sorted(unknown_telemetry)}; expected "
            "enabled, emit, callback, costs, dir, label, "
            "reset_day, period_start, period_end, budgets"
        )
    emit = str(telemetry_raw.get("emit", "log"))
    if emit not in VALID_EMIT:
        raise ConfigError(f"telemetry.emit must be one of {sorted(VALID_EMIT)}, got '{emit}'")
    callback = telemetry_raw.get("callback")
    if callback is not None and not isinstance(callback, str):
        raise ConfigError("telemetry.callback must be a string")
    log_dir = telemetry_raw.get("dir")
    if log_dir is not None and not isinstance(log_dir, str):
        raise ConfigError("telemetry.dir must be a string path")
    label = telemetry_raw.get("label")
    if label is not None and not isinstance(label, str):
        raise ConfigError("telemetry.label must be a string")
    telemetry = TelemetryConfig(
        enabled=bool(telemetry_raw.get("enabled", True)),
        emit=emit,
        callback=callback,
        costs=_parse_costs(telemetry_raw),
        dir=log_dir,
        label=label,
    )

    hooks_raw = dict(raw.get("hooks", {}))
    load = hooks_raw.get("load") or ()
    if isinstance(load, str):
        load = [load]
    if not all(isinstance(item, str) for item in load):
        raise ConfigError("hooks.load must be a list of module strings")
    hooks = HookConfig(load=tuple(load))

    billing = _parse_billing(telemetry_raw)

    return Config(
        gateway=gateway,
        providers=tuple(_parse_providers(raw, env)),
        telemetry=telemetry,
        billing=billing,
        hooks=hooks,
        models=_parse_models(raw),
        packs=_parse_packs(raw),
        source=str(path) if path else None,
        layers=(("embedded", str(path)) if path else ("embedded",)),
    )