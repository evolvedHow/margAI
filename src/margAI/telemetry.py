"""Per-call telemetry: usage, latency, and optional cost estimates.

Telemetry is opt-in per sink. Records flow through a :class:`Telemetry`
instance owned by the :class:`Wrapper`. The ``emit`` setting in config picks
one of ``none``, ``log``, or ``callback`` (a ``dotted.module:attr`` callable
taking a :class:`CallRecord`).
"""

from __future__ import annotations

import importlib
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import TelemetryConfig

__all__ = ["CallRecord", "Telemetry"]

logger = logging.getLogger("margAI.telemetry")


@dataclass
class CallRecord:
    source: str = "chat"
    request_model: str | None = None
    provider: str | None = None
    upstream_model: str | None = None
    stream: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    latency_ms: int | None = None
    status: int = 200
    error: str | None = None
    cost_usd: float | None = None
    # Which marglets the call activated, and why the router landed where it
    # did. Without these, a `dynamic` call is unattributable: the record would
    # show a provider nobody asked for and no hint that policy chose it.
    marglets: tuple[str, ...] = ()
    route_reason: str = ""
    created_at: float = field(default_factory=time.time)


class Telemetry:
    def __init__(self, config: TelemetryConfig, *, callback=None) -> None:
        self.config = config
        self._emit = config.emit
        self._callback = callback
        if config.emit == "callback" and self._callback is None:
            self._callback = _load_callable(config.callback)

    def record(self, **kwargs: Any) -> CallRecord:
        return CallRecord(**kwargs)

    def emit(self, record: CallRecord) -> None:
        if not self.config.enabled:
            return
        record.cost_usd = self.cost_for(record)
        try:
            if self._emit == "log":
                logger.info("%s", asdict(record))
            elif self._emit == "callback" and self._callback is not None:
                self._callback(record)
        except Exception:  # telemetry must never break the request path
            logger.exception("telemetry sink failed")

    def cost_for(self, record: CallRecord) -> float | None:
        if record.prompt_tokens is None or record.completion_tokens is None:
            return None
        if not self.config.costs:
            return None
        match = None
        for entry in self.config.costs:
            provider_ok = entry.provider in {record.provider, "*"}
            model_ok = entry.model in {record.upstream_model, "*"}
            if provider_ok and model_ok:
                match = entry
                break
        if match is None:
            return None
        return (
            record.prompt_tokens * match.input_price
            + record.completion_tokens * match.output_price
        ) / 1_000_000


def _load_callable(dotted: str | None):
    if not dotted:
        raise ValueError("telemetry.callback is required when emit='callback'")
    module_name, _, attr = dotted.partition(":")
    if not attr:
        module_name, _, attr = dotted.rpartition(".")
    module = importlib.import_module(module_name)
    for part in attr.split("."):
        module = getattr(module, part)
    return module