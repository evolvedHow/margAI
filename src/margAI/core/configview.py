"""Per-call configuration: what a pack may read, and what it may change.

A pack often needs configuration -- a prompt prefix, a threshold, a model
policy -- and the obvious places to put it are all wrong. A module global makes
two test cases share state. A constructor argument is invisible to the pack
author reading a tag handler. Reading a TOML file from inside a handler makes
routing depend on the working directory.

So: a :class:`ConfigView`, reachable from the request context, holding
immutable data that arrived with the call. Immutable, because two concurrent
calls must not be able to see each other's configuration.

The other half is deciding what a *caller* is allowed to send. A per-call
config is a remote input, so it gets a whitelist rather than the full config
grammar: callers may set ``[packs.<name>]`` for installed packs and
``[models.<name>]`` for virtual models, and nothing else. Everything else --
``hooks.load``, ``providers``, telemetry callbacks -- names a module to import
or an endpoint to reach, and a per-call config that could set those would turn
"send a request" into "run code on the gateway". :func:`safe_overlay` is the
chokepoint that draws that line, and it is enforced on every call rather than
trusted to callers to be well behaved.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CONFIG_KEY",
    "EMPTY_CONFIG",
    "ConfigView",
    "ConfigViolation",
    "deep_merge",
    "parse_overlay",
    "safe_overlay",
]

CONFIG_KEY = "margai.config"

#: Top-level tables a per-call config may contain. Everything outside this set
#: is rejected, not ignored: silently dropping a `hooks.load` the caller asked
#: for would let them believe a hook was installed.
SAFE_TABLES = frozenset({"packs", "models"})

#: Per-pack keys a caller may set. A pack's own namespace is data; a pack's
#: identity is not. Nothing here can name a module or a URL.
SAFE_PACK_KEYS = frozenset(
    {
        "enabled",
        "prefix",
        "suffix",
        "prompt",
        "instructions",
        "system",
        "threshold",
        "weight",
        "limit",
        "max",
        "min",
        "value",
        "default",
        "options",
        "label",
        "description",
    }
)

#: Guards the recursion below. TOML cannot nest deeply enough to need this;
#: a dict handed in by a Python caller can, and unbounded recursion on
#: attacker-shaped input is a stack overflow rather than an error.
MAX_DEPTH = 12


class ConfigViolation(ValueError):
    """A per-call config asked for something it may not have."""


@dataclass(frozen=True, slots=True)
class ConfigView:
    """Read-only per-call configuration for one request.

    Packs read it; they do not write it. Reaching a missing value returns
    ``None`` rather than raising, because a pack asking for configuration it
    was not given is a normal deployment, not an error -- a gateway with the
    pack installed but no config for it should serve traffic, not 500.
    """

    packs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    models: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    source: str = "default"

    def pack(self, name: str, key: str, default: Any = None) -> Any:
        """One value for one installed pack."""
        return self.packs.get(name, {}).get(key, default)

    def enabled(self, name: str, *, default: bool = True) -> bool:
        """Whether a pack is switched on for this call.

        Defaults to on: a pack is installed because the operator wanted it, so
        the caller's silence is not a request to disable it.
        """
        value = self.packs.get(name, {}).get("enabled", default)
        return bool(value)

    def model_policy(self, name: str) -> Mapping[str, Any]:
        """A caller-supplied override for one virtual model."""
        return self.models.get(name, {})

    def get(self, dotted: str, default: Any = None) -> Any:
        """Look up ``"packs.audit.threshold"``-style paths.

        Dotted lookup is what a pack author reaches for first, and a lookup
        helper here is cheaper than each pack writing its own four lines.
        """
        node: Any = {"packs": self.packs, "models": self.models}
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def as_dict(self) -> dict[str, Any]:
        return {"packs": dict(self.packs), "models": dict(self.models), "source": self.source}


#: The view every request starts from. A single shared instance is safe
#: because it is empty and frozen; allocating a new one per request would be
#: a small waste on the hot path for no behavioural difference.
EMPTY_CONFIG = ConfigView()


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any], _depth: int = 0) -> dict[str, Any]:
    """Merge ``overlay`` onto ``base``: tables recurse, everything else replaces.

    Tables merging and lists replacing is the asymmetry that matters. A caller
    sending ``prefix = "x"`` means "use x", and merging a list of prefixes into
    a configured one would produce a list where the pack expects a string --
    a type error surfacing as a 500 on a request the caller thought they had
    configured correctly.
    """
    if _depth > MAX_DEPTH:
        raise ConfigViolation("per-call config nests too deeply")
    out = dict(base)
    for key, value in overlay.items():
        current = out.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(current, value, _depth + 1)
        else:
            out[key] = value
    if _depth_of(out) > MAX_DEPTH:
        raise ConfigViolation("per-call config nests too deeply")
    return out


def _depth_of(value: Any, _depth: int = 1) -> int:
    """How deep the tables in ``value`` go.

    Counted on the *result*, not on the recursion: a caller can hand in an
    already-deep structure that never triggers a recursive ``deep_merge`` call,
    and a depth check that only counts its own recursion checks nothing.
    """
    if not isinstance(value, Mapping):
        return _depth
    return max((_depth_of(v, _depth + 1) for v in value.values()), default=_depth)


def _require_tables(raw: Any, where: str) -> dict[str, Mapping[str, Any]]:
    """Every entry under ``[packs]`` / ``[models]`` must itself be a table.

    Shape first, keys second: a caller who wrote ``[packs] audit = 3`` has
    made a different mistake from one who wrote ``[packs.audit] hooks = 1``,
    and conflating them would send them looking in the wrong place.
    """
    if not isinstance(raw, Mapping):
        raise ConfigViolation(f"[{where}] must be a table")
    out: dict[str, Mapping[str, Any]] = {}
    for name, value in raw.items():
        if not isinstance(value, Mapping):
            raise ConfigViolation(f"[{where}.{name}] must be a table")
        out[str(name)] = value
    return out


def _check_keys(table: Mapping[str, Any], where: str, allowed: frozenset[str]) -> None:
    """Reject keys that name a capability rather than a setting.

    Not recursive: everything under an allowed key is data (a list of options,
    a nested prompt object), not another layer of configuration. A value that
    names a module is a string in a list, and the gateway never imports
    anything a per-call config mentions.
    """
    for key in table:
        if key not in allowed:
            raise ConfigViolation(
                f"[{where}.{key}] is not settable per call. "
                f"Settable: {sorted(allowed) or 'nothing'}. "
                "Per-call config cannot name modules, providers or callbacks."
            )


def parse_overlay(text: str) -> dict[str, Any]:
    """Parse a per-call TOML string, rejecting unsafe tables at parse time.

    Accepts TOML or JSON: a JSON string is a valid subset for these shapes
    once quoted, and clients that build config programmatically should not
    need a TOML writer to send it.
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigViolation(f"per-call config is not valid TOML: {exc}") from exc
    return data


def safe_overlay(text: str | Mapping[str, Any]) -> ConfigView:
    """Turn a per-call config into a :class:`ConfigView`, or refuse it.

    The one entry point for untrusted per-call configuration. Everything it
    rejects is rejected with a message naming the key, because the caller is
    debugging their own request and "invalid config" with no location is a
    support ticket rather than a fix.
    """
    data = dict(text) if isinstance(text, Mapping) else parse_overlay(text)

    for top in data:
        if top not in SAFE_TABLES:
            raise ConfigViolation(
                f"[{top}] is not settable per call. Settable: {sorted(SAFE_TABLES)}. "
                "A per-call config can configure installed packs and virtual models; "
                "it cannot install code or reach an endpoint."
            )
    pack_tables = _require_tables(data.get("packs", {}), "packs")
    for name, table in pack_tables.items():
        _check_keys(table, f"packs.{name}", SAFE_PACK_KEYS)
    model_tables = _require_tables(data.get("models", {}), "models")
    return ConfigView(
        packs=_freeze(pack_tables),
        models=_freeze(model_tables),
        source="call",
    )


def _freeze(tables: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Copy nested tables into plain dicts.

    The view is described as immutable, and handing a pack the caller's actual
    nested dict would let it mutate a structure another pack is reading. A
    defensive copy at the boundary is cheaper than trusting every reader.
    """
    return {name: _freeze_any(dict(value)) for name, value in tables.items()}


def _freeze_any(value: Any, _depth: int = 0) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _freeze_any(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_freeze_any(item, _depth + 1) for item in value]
    return value


def iter_leaves(view: ConfigView) -> Iterator[tuple[str, Any]]:
    """Flatten a view to dotted paths, for logging a per-call config."""
    for table, entries in (("packs", view.packs), ("models", view.models)):
        for name, values in entries.items():
            for key, value in values.items():
                yield f"{table}.{name}.{key}", value
