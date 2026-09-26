"""Provider registry and construction from config."""

from __future__ import annotations

import os
from dataclasses import replace

from ..config import Config, ConfigError
from .anthropic import AnthropicProvider
from .openai_compat import OpenAICompatProvider, Provider

__all__ = ["PROVIDER_TYPES", "Provider", "build_providers"]

# kind -> provider class.
PROVIDER_TYPES: dict[str, type[Provider]] = {
    "openai": OpenAICompatProvider,
    "openai-compatible": OpenAICompatProvider,
    "anthropic": AnthropicProvider,
}


def build_providers(config: Config, env: dict[str, str] | None = None) -> dict[str, Provider]:
    """Instantiate (and validate) providers from a :class:`Config`.

    ``api_key_env`` names are resolved against ``env`` (defaults to
    ``os.environ``) at build time, so auth headers are populated from the
    environment as the config documents.
    """
    env = dict(os.environ if env is None else env)
    providers: dict[str, Provider] = {}
    for cfg in config.providers:
        provider_cls = PROVIDER_TYPES.get(cfg.kind)
        if provider_cls is None:
            raise ConfigError(
                f"provider '{cfg.name}': unknown kind '{cfg.kind}' "
                f"(supported: {', '.join(sorted(PROVIDER_TYPES))})"
            )
        if cfg.api_key is None and cfg.api_key_env is not None and not cfg.api_key_env:
            raise ConfigError(f"provider '{cfg.name}': api_key_env must not be empty")
        key = cfg.resolved_key(env)
        if key is not None and key != cfg.api_key:
            cfg = replace(cfg, api_key=key)
        providers[cfg.name] = provider_cls(cfg)
    return providers