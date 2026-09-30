"""Pure-stdlib core of margAI.

Nothing in :mod:`margAI.core` touches the network or depends on external
frameworks. Interceptors, routing, and the transport *contract* live here so
the same core can be embedded behind FastAPI, a Modal endpoint, or a
Cloudflare Worker (which supplies its own transport implementation).
"""

from .billing import BudgetTable, SpendLedger
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
from .tags import (
    RESERVED_NAMESPACE,
    ROOT_PACK,
    TagConflictError,
    TagHandler,
    TagRegistry,
    namespace_key,
    qualified_key,
    qualify,
)

__all__ = [
    "DONE",
    "RESERVED_NAMESPACE",
    "ROOT_PACK",
    "ApiError",
    "BudgetTable",
    "DoneSentinel",
    "EventHandler",
    "EventRegistry",
    "HookKind",
    "HookRegistry",
    "ModelRouter",
    "PreparedRequest",
    "RequestContext",
    "Route",
    "SpendLedger",
    "TagConflictError",
    "TagHandler",
    "TagRegistry",
    "Transport",
    "UpstreamResponse",
    "UpstreamStream",
    "maybe_await",
    "namespace_key",
    "qualified_key",
    "qualify",
]