"""Provider registry and construction from config."""

from __future__ import annotations

from ..config import Config, ConfigError
from .openai_compat import OpenAICompatProvider

__all__ = ["PROVIDER_TYPES", "build_providers"]

# kind -> provider class. Add AnthropicProvider here when it exists.
PROVIDER_TYPES: dict[str, type[OpenAICompatProvider]] = {
    "openai": OpenAICompatProvider,
    "openai-compatible": OpenAICompatProvider,
}


def build_providers(config: Config) -> dict[str, OpenAICompatProvider]:
    """Instantiate (and validate) providers from a :class:`Config`."""
    providers: dict[str, OpenAICompatProvider] = {}
    for cfg in config.providers:
        provider_cls = PROVIDER_TYPES.get(cfg.kind)
        if provider_cls is None:
            raise ConfigError(
                f"provider '{cfg.name}': unknown kind '{cfg.kind}' "
                f"(supported: {', '.join(sorted(PROVIDER_TYPES))})"
            )
        if cfg.api_key is None and cfg.api_key_env is not None and not cfg.api_key_env:
            raise ConfigError(f"provider '{cfg.name}': api_key_env must not be empty")
        providers[cfg.name] = provider_cls(cfg)
    return providers