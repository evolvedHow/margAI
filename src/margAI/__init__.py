"""margAI -- portable LLM wrapper framework.

A wrapper is a reusable layer of interceptors around any upstream LLM
(OpenAI, OpenRouter, vLLM, Ollama, ...). Expose it locally or on Modal behind
the OpenAI-compatible FastAPI adapter, or port the pure-stdlib core to a
Cloudflare Worker with a custom transport.

Quick start:

    from margAI import Wrapper, build_app
    from margAI.config import load_config

    app = build_app(config=load_config())

Or via the CLI: ``uv run margAI`` (config from ``margAI.toml`` or
``MARGAI_CONFIG``).
"""

from .bangtag import Tag, find_directive, install_bangtags, remove_directive
from .config import Config, ConfigError, load_config
from .core import (
    DONE,
    ApiError,
    EventHandler,
    EventRegistry,
    HookKind,
    HookRegistry,
    ModelRouter,
    PreparedRequest,
    RequestContext,
    Route,
    Transport,
    UpstreamResponse,
    UpstreamStream,
    maybe_await,
)
from .core.intent import RoutingIntent
from .core.marglets import Marglet, MargletRegistry, MargletSpec
from .core.router import Candidate
from .providers import build_providers
from .routing import Routing, Selector, coerce_route
from .telemetry import CallRecord, Telemetry
from .transport.fastapi import build_app
from .wrapper import GatewayResponse, StreamHandle, Wrapper

__version__ = "0.1.0"

__all__ = [
    "DONE",
    "ApiError",
    "CallRecord",
    "Candidate",
    "Config",
    "ConfigError",
    "EventHandler",
    "EventRegistry",
    "GatewayResponse",
    "HookKind",
    "HookRegistry",
    "Marglet",
    "MargletRegistry",
    "MargletSpec",
    "ModelRouter",
    "PreparedRequest",
    "RequestContext",
    "Route",
    "Routing",
    "RoutingIntent",
    "Selector",
    "StreamHandle",
    "Tag",
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
    "install_bangtags",
    "load_config",
    "maybe_await",
    "remove_directive",
]