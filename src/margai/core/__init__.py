"""Pure-stdlib core of margai.

Nothing in :mod:`margai.core` touches the network or depends on external
frameworks. Interceptors, routing, and the transport *contract* live here so
the same core can be embedded behind FastAPI, a Modal endpoint, or a
Cloudflare Worker (which supplies its own transport implementation).
"""

from .context import RequestContext
from .errors import DONE, ApiError
from .hooks import HookKind, HookRegistry, maybe_await
from .protocol import (
    PreparedRequest,
    Transport,
    UpstreamResponse,
    UpstreamStream,
)
from .router import ModelRouter, Route

__all__ = [
    "ApiError",
    "DONE",
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