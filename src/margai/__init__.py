"""margai -- portable LLM wrapper framework.

A wrapper is a reusable layer of interceptors around any upstream LLM
(OpenAI, OpenRouter, vLLM, Ollama, ...). Expose it locally or on Modal behind
the OpenAI-compatible FastAPI adapter, or port the pure-stdlib core to a
Cloudflare Worker with a custom transport.

Quick start:

    from margai import Wrapper, build_app
    from margai.config import load_config

    app = build_app(config=load_config())

Or via the CLI: ``uv run margai`` (config from ``margai.toml`` or
``MARGAI_CONFIG``).
"""

from .config import Config, ConfigError, load_config
from .core import (
    DONE,
    ApiError,
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
from .providers import build_providers
from .telemetry import CallRecord, Telemetry
from .transport.fastapi import build_app
from .wrapper import GatewayResponse, StreamHandle, Wrapper

__version__ = "0.1.0"

__all__ = [
    "ApiError",
    "CallRecord",
    "Config",
    "ConfigError",
    "DONE",
    "GatewayResponse",
    "HookKind",
    "HookRegistry",
    "ModelRouter",
    "PreparedRequest",
    "RequestContext",
    "Route",
    "StreamHandle",
    "Telemetry",
    "Transport",
    "UpstreamResponse",
    "UpstreamStream",
    "Wrapper",
    "__version__",
    "build_app",
    "build_providers",
    "load_config",
    "maybe_await",
]