"""``margAI doctor`` -- why is my tag not doing anything?

The failure this exists for is silent. A tag whose pack is not installed, a
bangtag in a namespace the gateway never claimed, a virtual model nobody
configured: each of them produces a successful request and a prompt that went
out unchanged. Nothing in a log says so. This is the command that says it.

Every check reads the live gateway -- the tag registry, the router, the
resolved config -- rather than re-parsing the config file, so a check can
never pass because the config looked right while the wiring is wrong.

Exit status is 0 when nothing is wrong and 1 when something is, so it works in
CI and in a container healthcheck without anyone having to read the output.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TextIO

from .config import ConfigError, load_config

__all__ = ["Check", "Doctor", "run_doctor"]

# Findings below this are advice, not faults: they do not fail the exit code.
ADVICE = "advice"
WARN = "warn"
ERROR = "error"
OK = "ok"


@dataclass(slots=True)
class Check:
    """One finding. ``level`` decides the exit code, not the wording."""

    level: str
    subject: str
    message: str
    hint: str = ""

    def render(self) -> str:
        mark = {OK: "ok  ", WARN: "WARN", ERROR: "FAIL", ADVICE: "note"}[self.level]
        line = f"  {mark}  {self.subject}: {self.message}"
        return f"{line}\n        {self.hint}" if self.hint else line


@dataclass(slots=True)
class Doctor:
    """Collects checks so the caller can render or serialise them."""

    checks: list[Check] = field(default_factory=list)

    def add(self, level: str, subject: str, message: str, hint: str = "") -> None:
        self.checks.append(Check(level, subject, message, hint))

    @property
    def failed(self) -> bool:
        return any(c.level == ERROR for c in self.checks)

    def render(self, out: TextIO) -> None:
        for check in self.checks:
            print(check.render(), file=out)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": not self.failed,
            "checks": [
                {"level": c.level, "subject": c.subject, "message": c.message, "hint": c.hint}
                for c in self.checks
            ],
        }


def run_doctor(
    *,
    source: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
    out: TextIO | None = None,
    render: bool = True,
) -> Doctor:
    """Run every check against a gateway built from ``source``.

    ``render=False`` collects without printing, for the ``--json`` path --
    mixing a human report and a JSON document on one stream produces output
    neither a person nor a parser can read.
    """
    import sys

    out = out or sys.stdout
    env = dict(os.environ if env is None else env)
    doctor = Doctor()

    try:
        config = load_config(source, env=env)
    except ConfigError as exc:
        doctor.add(ERROR, "config", str(exc), "Fix the file, or run without -c to use margAI.toml.")
        if render:
            doctor.render(out)
        return doctor

    _check_config(doctor, config)
    wrapper = _build(config)
    _check_packs(doctor, wrapper)
    _check_tags(doctor, wrapper)
    _check_providers(doctor, wrapper, env)
    _check_virtual_models(doctor, wrapper)
    _check_routing(doctor, wrapper)
    if render:
        doctor.render(out)
    return doctor


class _Offline:
    """A transport that refuses, so a doctor run can never send anything.

    The doctor inspects wiring, not upstreams. Dialling out to check
    reachability would make the command slow, flaky, and surprising to run in
    an offline CI job -- and a diagnostic that has its own side effects is a
    diagnostic nobody trusts.
    """

    async def request(self, req: Any) -> Any:
        raise RuntimeError("margAI doctor does not send requests")

    async def open_stream(self, req: Any) -> Any:
        raise RuntimeError("margAI doctor does not send requests")


def _build(config: Any) -> Any:
    """Build the gateway exactly the way `serve` builds it.

    Via ``from_config`` on purpose. Constructing a Wrapper by hand here would
    leave ``default_provider``, ``prefix`` and the rest at their defaults, and
    every check that reads them would then be reporting on a gateway that is
    not the one the operator runs -- which is worse than not reporting at all.
    """
    from .wrapper import Wrapper

    return Wrapper.from_config(config, transport=_Offline(), load_hooks=False)


# -- checks -----------------------------------------------------------------


def _check_config(doctor: Doctor, config: Any) -> None:
    if config.source is None:
        doctor.add(
            ADVICE,
            "config",
            f"no config file; using the embedded defaults ({', '.join(config.layers)})",
            "Copy margAI.toml and edit it to pin providers and keys.",
        )
    else:
        doctor.add(OK, "config", f"{config.source} over {' over '.join(config.layers[:-1]) or 'defaults'}")

    if not config.providers:
        doctor.add(ERROR, "config", "no providers configured", "Nothing can route. Add a [providers.*] table.")

    for provider in config.providers:
        if provider.models or provider.default_model is not None:
            continue
        level = (
            # The severity turns on what the provider is *for*. A model-less
            # provider that nothing defaults to is dead weight worth a warning;
            # the same provider behind gateway.default_provider is the only
            # thing an unqualified request can land on, so every unqualified
            # request fails and that is an error, not a warning.
            ERROR
            if provider.name == config.gateway.default_provider
            else WARN
        )
        if level is ERROR:
            message = "no models and no default_model, but it is gateway.default_provider"
            hint = "Give it a `models` list, or point gateway.default_provider elsewhere."
        else:
            message = "no models and no default_model"
            hint = "Clients cannot discover anything on it; requests naming it will fail."
        doctor.add(level, f"provider {provider.name}", message, hint)


def _check_packs(doctor: Doctor, wrapper: Any) -> None:
    for failure in wrapper.pack_failures:
        doctor.add(
            ERROR,
            f"pack {failure.name}",
            f"failed to install: {failure.reason}",
            f"Check `{failure.target or 'its entry point'}` and the gateway log.",
        )
    for record in getattr(wrapper, "packs", ()):
        if record.installed and not record.tags:
            doctor.add(
                WARN,
                f"pack {record.name}",
                "installed but registered no tags",
                "Either it only adds hooks, or it silently did nothing you expected.",
            )
    if not wrapper.pack_failures and not getattr(wrapper, "packs", ()):
        doctor.add(
            ADVICE,
            "packs",
            "no third-party packs discovered",
            "Install one, or set [gateway] pack_discovery = false.",
        )


def _check_tags(doctor: Doctor, wrapper: Any) -> None:
    described = wrapper.tags.describe()
    if not described:
        doctor.add(ERROR, "tags", "no tags registered at all", "The built-in pack did not install.")
    for namespace, entries in sorted(described.items()):
        for entry in entries:
            if not entry["doc"]:
                doctor.add(
                    ADVICE,
                    f"{namespace}:{entry['name']}",
                    "no docstring, so it cannot be discovered",
                    "The first line of the handler's docstring is what /v1/margAI/tags reports.",
                )
    policy = wrapper.config.gateway.on_unknown_tag if wrapper.config else "warn"
    if policy == "warn":
        doctor.add(
            ADVICE,
            "tags",
            "on_unknown_tag = warn",
            "A typo in a namespace is only a log line. Set it to `error` in production.",
        )


def _check_providers(doctor: Doctor, wrapper: Any, env: dict[str, str]) -> None:
    for name, provider in sorted(wrapper.providers.items()):
        config = getattr(provider, "config", None)
        if config is None:
            continue
        if config.resolved_key(env):
            continue
        if config.api_key_env:
            # The one that is definitely a problem: the config asked for a key
            # and the environment does not have it.
            doctor.add(
                WARN,
                f"provider {name}",
                f"no API key (expected ${config.api_key_env})",
                "Requests will fail upstream until it is set.",
            )
        elif config.api_key:
            doctor.add(ERROR, f"provider {name}", "api_key set but empty", "Remove it, or set the variable it names.")
        else:
            # Neither declared. That is a deliberate local gateway (llama.cpp,
            # vLLM on a LAN) as often as it is a forgotten line, so it is a
            # note and not a warning -- warning on every keyless local provider
            # trains people to ignore the warning that matters.
            doctor.add(
                ADVICE,
                f"provider {name}",
                "no auth configured",
                "Fine for a local upstream. Set api_key_env if it needs one.",
            )
    routed = {c.provider for c in wrapper.router.candidates()}
    for name in wrapper.providers:
        if name not in routed:
            doctor.add(
                WARN,
                f"provider {name}",
                "offers no configured models, so routing can never pick it",
                "Add a `models` list, or remove the provider.",
            )


def _check_virtual_models(doctor: Doctor, wrapper: Any) -> None:
    for entry in wrapper.virtuals.describe():
        subject = entry["id"]
        if entry["max_cost_per_1m"] is not None and not wrapper.virtuals.costs:
            doctor.add(
                ERROR,
                subject,
                "has a cost ceiling but no prices are configured",
                "No candidate can be shown to qualify, so it will 404. Add [telemetry.costs].",
            )
        pool = [
            c
            for c in wrapper.router.candidates()
            if not (entry["providers"] and c.provider not in entry["providers"])
            and c.provider not in entry["exclude_providers"]
        ]
        if not pool:
            doctor.add(
                ERROR,
                subject,
                "its filters exclude every configured model",
                "It can never resolve. Relax the filters or add a matching model.",
            )
        if entry["prefer"]:
            known = {f"{c.provider}/{c.model}" for c in wrapper.router.candidates()}
            missing = [p for p in entry["prefer"] if p not in known]
            if missing:
                doctor.add(
                    ADVICE,
                    subject,
                    f"prefers {missing}, which no configured model matches",
                    "Preferences that match nothing are skipped, not fatal.",
                )


def _check_routing(doctor: Doctor, wrapper: Any) -> None:
    if not wrapper.router.candidates():
        doctor.add(ERROR, "routing", "no routable (provider, model) pairs", "Nothing can be routed, qualified or bare.")
    if wrapper.router.default_provider and wrapper.router.default_provider not in wrapper.providers:
        doctor.add(
            ERROR,
            "routing",
            f"default_provider '{wrapper.router.default_provider}' is not configured",
            "Every unqualified model would fail. Fix the name or remove it.",
        )
    if not wrapper.routing.selectors():
        doctor.add(
            ADVICE,
            "routing",
            f"no selectors registered, so '{wrapper.router.dynamic_id}' can only fall back",
            "Add wrapper.routing.add_selector(...) or set gateway.default_provider.",
        )


def main(argv: Sequence[str] | None = None) -> int:
    """``margAI doctor [-c FILE] [--json]``."""
    import argparse

    parser = argparse.ArgumentParser(prog="margAI doctor", description="Diagnose a margAI gateway's wiring.")
    parser.add_argument("-c", "--config", default=None, help="config file to check (default: margAI.toml)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logs too")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    if args.json:
        result = run_doctor(source=args.config, render=False)
        print(json.dumps(result.as_dict(), indent=2))
    else:
        result = run_doctor(source=args.config)
    return 1 if result.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
