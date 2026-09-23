"""Model routing: ``<prefix>/<provider>/<model>`` namespacing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .errors import ApiError

__all__ = ["ModelRouter", "Route"]


@dataclass(frozen=True)
class Route:
    provider: str
    model: str


class ModelRouter:
    """Resolve a client-supplied model id to a ``(provider, upstream model)`` pair.

    Resolution is offline: it uses each provider's *configured* ``models``
    list (the source of truth) plus provider-qualified segments. Network
    model discovery is only used to populate the ``/v1/models`` catalog, so
    the hot path never blocks on upstream.
    """

    def __init__(
        self,
        providers: dict[str, Any],
        prefix: str = "marg",
        expose: str = "prefixed",
        default_provider: str | None = None,
    ) -> None:
        self.providers = providers
        self.prefix = prefix
        self.expose = expose
        self.default_provider = default_provider

    # -- id generation ------------------------------------------------------

    def namespaced_id(self, provider: str, model: str) -> str:
        return f"{self.prefix}/{provider}/{model}"

    def split_id(self, model_id: str) -> tuple[str, str] | None:
        """Return ``(provider, model)`` if ``model_id`` starts with the prefix."""
        prefix = self.prefix + "/"
        if not model_id.startswith(prefix):
            return None
        rest = model_id[len(prefix) :]
        provider, sep, model = rest.partition("/")
        return (provider, model if sep else "")

    # -- resolution ---------------------------------------------------------

    def resolve(self, model_id: str | None) -> Route:
        prefix = self.prefix + "/"

        if not model_id:
            if self.default_provider:
                return Route(self.default_provider, _provider_default_model(self, self.default_provider))
            raise ApiError(400, "No model specified", error_type="invalid_request_error", param="model")

        if model_id.startswith(prefix):
            rest = model_id[len(prefix) :]
            provider, sep, model = rest.partition("/")
            if provider in self.providers:
                return Route(provider, model or _provider_default_model(self, provider))
            # Not a configured provider: the whole token might be a model id
            # that itself contains a slash. Try bare lookup, then give up with
            # a helpful provider hint.
            try:
                return self._resolve_bare(rest, origin=model_id)
            except ApiError:
                if sep and provider:
                    raise ApiError(
                        404,
                        f"Unknown provider '{provider}' in model id '{model_id}'. "
                        f"Configured providers: {', '.join(sorted(self.providers)) or 'none'}.",
                        error_type="invalid_request_error",
                        param="model",
                    ) from None
                raise

        if self.expose == "prefixed":
            raise ApiError(
                404,
                f"Unknown model '{model_id}'. Use {prefix}<provider>/<model> "
                f"(see /v1/models for the available namespaced ids).",
                error_type="invalid_request_error",
                param="model",
            )
        return self._resolve_bare(model_id, origin=model_id)

    def _resolve_bare(self, model: str, origin: str) -> Route:
        matches = [name for name, p in self.providers.items() if model in set(p.configured_models())]
        if len(matches) == 1:
            return Route(matches[0], model)
        if len(matches) > 1:
            if self.default_provider in matches:
                return Route(self.default_provider, model)
            raise ApiError(
                400,
                f"Model '{model}' exists on multiple providers: {sorted(matches)}. "
                f"Qualify it as '{self.prefix}/<provider>/{model}'.",
                error_type="invalid_request_error",
                param="model",
            )
        if self.default_provider:
            return Route(self.default_provider, model)
        raise ApiError(
            404,
            f"Unknown model '{origin}'. See /v1/models for available ids.",
            error_type="invalid_request_error",
            param="model",
        )

    # -- catalog ------------------------------------------------------------

    def catalog(self, provider_models: dict[str, list[str]]) -> list[dict[str, Any]]:
        """Build the OpenAI-shaped ``/v1/models`` payload from per-provider
        model id lists, honoring the ``expose`` mode.

        ``prefixed`` exposes only ``<prefix>/<provider>/<model>`` ids;
        ``both`` additionally exposes the bare id the first time it appears;
        ``raw`` exposes only bare ids.
        """
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        now = int(time.time())

        def add(ident: str, *, parent: str, owned_by: str) -> None:
            if ident in seen:
                return
            seen.add(ident)
            items.append(
                {"id": ident, "object": "model", "created": now, "owned_by": owned_by, "parent": parent}
            )

        for provider, models in provider_models.items():
            for model in models:
                if self.expose != "raw":
                    add(self.namespaced_id(provider, model), parent=model, owned_by=provider)
                if self.expose in ("both", "raw"):
                    add(model, parent=model, owned_by=provider)
        return items


def _provider_default_model(router: ModelRouter, provider: str) -> str:
    prov = router.providers.get(provider)
    default = getattr(prov, "config", None) and getattr(prov.config, "default_model", None)
    if default:
        return default
    raise ApiError(
        404,
        f"Provider '{provider}' has no default_model configured.",
        error_type="invalid_request_error",
        param="model",
    )