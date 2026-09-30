"""Per-call telemetry: usage, latency, and optional cost estimates.

Telemetry is opt-in per sink. Records flow through a :class:`Telemetry`
instance owned by the :class:`Wrapper`. The ``emit`` setting in config picks
one of ``none``, ``log``, ``callback`` (a ``dotted.module:attr`` callable
taking a :class:`CallRecord`), or ``file``.

``emit = "file"`` appends one JSON object per line to a log whose name is
generated from the app and the start time -- ``vedanta_telemetry_260929143022``
-- under a standard per-user directory unless ``[telemetry] dir`` names one.
The file is opened once, at construction, so a gateway that has been running
for days keeps writing to the file it started with rather than a new one.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import TelemetryConfig

__all__ = ["CallRecord", "Telemetry"]

logger = logging.getLogger("margAI.telemetry")


@dataclass(slots=True)
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
    def __init__(
        self,
        config: TelemetryConfig,
        *,
        callback: Callable[[CallRecord], None] | None = None,
        label: str | None = None,
    ) -> None:
        self.config = config
        self._emit = config.emit
        self._callback = callback
        #: Path of the ``emit = "file"`` sink, once one is open.
        self.path: Path | None = None
        self._file: Any = None
        if config.emit == "callback" and self._callback is None:
            self._callback = _load_callable(config.callback)
        if config.emit == "file" and config.enabled:
            self.path, self._file = _open_log_file(config, label)

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
            elif self._emit == "file" and self._file is not None:
                # One JSON object per line: appendable, greppable, and
                # re-readable with no schema. Flushed per record so a hard exit
                # loses at most the line in flight -- telemetry is only useful
                # if it survived the crash you are trying to debug.
                self._file.write(json.dumps(asdict(record), default=str) + "\n")
                self._file.flush()
        except Exception:  # telemetry must never break the request path
            logger.exception("telemetry sink failed")

    def close(self) -> None:
        """Close the file sink, if any. Safe to call more than once."""
        if self._file is not None:
            try:
                self._file.close()
            finally:
                self._file = None

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


def _default_log_dir() -> Path:
    """Standard per-user telemetry directory, XDG-aware."""
    state = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(state) / "margAI" / "logs"


def _slug(text: str) -> str:
    """Filesystem-safe fragment for the generated file name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return cleaned or "margAI"


def _open_log_file(config: TelemetryConfig, label: str | None) -> tuple[Path, Any]:
    """Open the generated telemetry log for appending.

    Name is ``{label}_telemetry_{yymmddhhmmss}.log``; the local start time is
    part of the name so successive runs never overwrite each other, and the
    label (gateway prefix by default) is what distinguishes one app's log from
    another sharing the directory.
    """
    directory = Path(config.dir).expanduser() if config.dir else _default_log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%y%m%d%H%M%S")
    path = directory / f"{_slug(config.label or label or 'margAI')}_telemetry_{stamp}.log"
    logger.info("telemetry log: %s", path)
    return path, path.open("a", encoding="utf-8")


def _load_callable(dotted: str | None) -> Any:
    if not dotted:
        raise ValueError("telemetry.callback is required when emit='callback'")
    module_name, _, attr = dotted.partition(":")
    if not attr:
        module_name, _, attr = dotted.rpartition(".")
    module = importlib.import_module(module_name)
    for part in attr.split("."):
        module = getattr(module, part)
    return module