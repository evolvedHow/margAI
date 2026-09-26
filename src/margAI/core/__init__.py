"""Pure-stdlib core of margAI.

Nothing in :mod:`margAI.core` touches the network or depends on external
frameworks. Interceptors, routing, and the transport *contract* live here so
the same core can be embedded behind FastAPI, a Modal endpoint, or a
Cloudflare Worker (which supplies its own transport implementation).
"""

from .context import RequestContext
from .errors import DONE, ApiError, DoneSentinel
from .events import EventHandler, EventRegistry
from .hooks import HookKind, HookRegistry, maybe_await
from .protocol import (
    PreparedRequest,
    Transport,
    UpstreamResponse,
    UpstreamStream,
)
from .router import ModelRouter, Route

__all__ = [
    "DONE",
    "ApiError",
    "DoneSentinel",
    "EventHandler",
    "EventRegistry",
    "HookKind",
    "HookRegistry",
    "ModelRouter",
    "PreparedRequest",
    "RequestContext",
    "Route",
    "Transport",
    "UpstreamResponse",
    "UpstreamStream",
    "maybe_await",
]