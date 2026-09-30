"""margAI -- portable LLM wrapper framework.

A wrapper is a reusable layer of interceptors around any upstream LLM
(OpenAI, OpenRouter, vLLM, Ollama, ...). Expose it locally or on Modal behind
the OpenAI-compatible FastAPI adapter, or port the pure-stdlib core to a
Cloudflare Worker with a custom transport.

Quick start (simplified Gateway API):

    from margAI import Gateway

    app = Gateway()  # Reads margAI.toml automatically

    @app.tag("summarize")
    def summarize(ctx, tag):
        ctx.add_system("Provide a brief summary.")

    app.serve()

Or use the full Wrapper API:

    from margAI import Wrapper, build_app
    from margAI.config import load_config

    app = build_app(config=load_config())

Or via the CLI: ``uv run margAI`` (config from ``margAI.toml`` or
``MARGAI_CONFIG``).
"""

from .bangtag import (
    BANGTAG_NS,
    Directive,
    Tag,
    find_directive,
    find_directives,
    install_bangtags,
    remove_directive,
    strip_directives,
)
from .config import Config, ConfigError, load_config
from .core import (
    DONE,
    ApiError,
    DoneSentinel,
    EventHandler,
    EventRegistry,
    HookKind,
    HookRegistry,
    ModelRouter,
    PreparedRequest,
    RequestContext,
    Route,
    TagConflictError,
    TagHandler,
    TagRegistry,
    Transport,
    UpstreamResponse,
    UpstreamStream,
    maybe_await,
)
from .core.context import SCRATCH_KEY, STATE_KEY
from .core.intent import RoutingIntent
from .core.marglets import Marglet, MargletRegistry, MargletSpec
from .core.responses import ErrorResponse, ImmediateResponse, RouteOverride
from .core.router import Candidate
from .gateway import Gateway
from .providers import build_providers
from .routing import Routing, Selector, coerce_route
from .telemetry import CallRecord, Telemetry
from .transport.fastapi import build_app
from .wrapper import GatewayResponse, StreamHandle, Wrapper

__version__ = "0.1.0"

__all__ = [
    "BANGTAG_NS",
    "DONE",
    "SCRATCH_KEY",
    "STATE_KEY",
    "ApiError",
    "CallRecord",
    "Candidate",
    "Config",
    "ConfigError",
    "Directive",
    "DoneSentinel",
    "ErrorResponse",
    "EventHandler",
    "EventRegistry",
    "Gateway",  # Simplified API
    "GatewayResponse",
    "HookKind",
    "HookRegistry",
    "ImmediateResponse",
    "Marglet",
    "MargletRegistry",
    "MargletSpec",
    "ModelRouter",
    "PreparedRequest",
    "RequestContext",
    "Route",
    "RouteOverride",
    "Routing",
    "RoutingIntent",
    "Selector",
    "StreamHandle",
    "Tag",
    "TagConflictError",
    "TagHandler",
    "TagRegistry",
    "Telemetry",
    "Transport",
    "UpstreamResponse",
    "UpstreamStream",
    "Wrapper",
    "__version__",
    "build_app",
    "build_providers",
    "coerce_route",
    "find_directive",
    "find_directives",
    "install_bangtags",
    "load_config",
    "maybe_await",
    "remove_directive",
    "strip_directives",
]
