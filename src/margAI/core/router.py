"""Model routing: ``<prefix>/<provider>/<model>`` namespacing.

Two resolution paths:

- **qualified** -- ``margAI/openai/gpt-4o``. Always unambiguous.
- **bare** -- ``gpt-4o``, resolved against every provider's configured model
  list, and only offered when ``expose`` is not ``prefixed``.

``default_provider`` is the fallback for both: it catches a bare model id
nobody claims and a request that names no model at all. ``expose`` controls
what ``/v1/models`` *lists*; it never gates what the router will *accept*.
Keeping those two concerns separate is deliberate -- conflating them meant a
config with ``expose = "prefixed"`` silently rejected every bare model id
even with a ``default_provider`` set, and told the operator so nowhere.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .errors import ApiError

__all__ = ["Candidate", "ModelRouter", "Route"]

logger = logging.getLogger("margAI.router")


@dataclass(frozen=True)
class Route:
    provider: str
    model: str
    reason: str = ""


@dataclass(frozen=True)
class Candidate:
    """One routable (provider, model) pair offered to the dynamic router."""

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
        prefix: str = "margAI",
        expose: str = "prefixed",
        default_provider: str | None = None,
        dynamic_model: str = "dynamic",
    ) -> None:
        self.providers = providers
        self.prefix = prefix
        self.expose = expose
        self.default_provider = default_provider
        self.dynamic_model = dynamic_model
        self._dynamic_id = f"{prefix}/{dynamic_model}"
        # Reverse index: model id -> providers offering it. Rebuilt on
        # invalidate(); resolution is on every request, so it must not
        # re-derive this per call.
        self._by_model: dict[str, list[str]] = {}
        self.invalidate()

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

    @property
    def dynamic_id(self) -> str:
        """The reserved model id whose target is chosen per call."""
        return self._dynamic_id

    def is_dynamic(self, model_id: str | None) -> bool:
        """True for the reserved dynamic id. The bare form only counts when
        no provider actually offers a model by that name."""
        if not model_id:
            return False
        if model_id == self._dynamic_id:
            return True
        return model_id == self.dynamic_model and model_id not in self._by_model

    def invalidate(self) -> None:
        """Rebuild the bare-model index from the providers' configured lists."""
        index: dict[str, list[str]] = {}
        for name, provider in self.providers.items():
            for model in provider.configured_models():
                index.setdefault(model, []).append(name)
        self._by_model = index

    def candidates(self) -> list[Candidate]:
        """Every routable (provider, model) pair, in a stable order."""
        return [
            Candidate(provider=provider, model=model)
            for model in sorted(self._by_model)
            for provider in sorted(self._by_model[model])
        ]

    # -- resolution ---------------------------------------------------------

    def resolve(self, model_id: str | None) -> Route:
        """Resolve a client model id. Raises :class:`ApiError` if it cannot."""
        route = self.try_resolve(model_id)
        if route is None:
            raise self.explain(model_id)
        return route

    def try_resolve(self, model_id: str | None) -> Route | None:
        """Like :meth:`resolve` but returns ``None`` instead of raising.

        Used for the dynamic path, where "cannot resolve" is a normal outcome
        that falls through to the selector chain rather than an error.
        """
        prefix = self.prefix + "/"

        if not model_id:
            if not self.default_provider:
                return None
            return Route(
                self.default_provider,
                _provider_default_model(self, self.default_provider),
                "default_provider",
            )

        if model_id.startswith(prefix):
            rest = model_id[len(prefix) :]
            provider, sep, model = rest.partition("/")
            if provider in self.providers:
                if not model:
                    return Route(
                        provider,
                        _provider_default_model(self, provider),
                        "provider_default",
                    )
                return Route(provider, model, "qualified")
            # Not a configured provider: the whole token might be a model id
            # that itself contains a slash. Try bare lookup.
            route = self._resolve_bare(rest)
            if route is not None:
                return route
            if sep and provider:
                raise ApiError(
                    404,
                    f"Unknown provider '{provider}' in model id '{model_id}'. "
                    f"Configured providers: {', '.join(sorted(self.providers)) or 'none'}.",
                    error_type="invalid_request_error",
                    param="model",
                )
            return None

        route = self._resolve_bare(model_id)
        if route is not None:
            return route
        if self.default_provider:
            return Route(self.default_provider, model_id, "default_provider")
        return None

    def _resolve_bare(self, model: str) -> Route | None:
        matches = self._by_model.get(model, ())
        if len(matches) == 1:
            return Route(matches[0], model, "bare")
        if len(matches) > 1:
            if self.default_provider in matches:
                logger.warning(
                    "model %r is offered by %s; routing to default_provider %r. "
                    "Qualify it as %s to choose explicitly.",
                    model,
                    sorted(matches),
                    self.default_provider,
                    self.namespaced_id(self.default_provider, model),
                )
                return Route(self.default_provider, model, "default_provider")
            raise ApiError(
                400,
                f"Model '{model}' exists on multiple providers: {sorted(matches)}. "
                f"Qualify it as '{self.prefix}/<provider>/{model}'.",
                error_type="invalid_request_error",
                param="model",
            )
        return None

    def explain(self, model_id: str | None) -> ApiError:
        """The error for a model id that would not resolve."""
        if not model_id:
            return ApiError(
                400,
                "No model specified",
                error_type="invalid_request_error",
                param="model",
            )
        configured = ", ".join(sorted(self.providers)) or "none"
        return ApiError(
            404,
            f"Unknown model '{model_id}'. Use {self.prefix}/<provider>/<model> "
            f"(configured providers: {configured}; see /v1/models for the available ids).",
            error_type="invalid_request_error",
            param="model",
        )

    # -- catalog ------------------------------------------------------------

    def catalog(self, provider_models: dict[str, list[str]], *, include_dynamic: bool = False) -> list[dict[str, Any]]:
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

        if include_dynamic:
            add(self._dynamic_id, parent=self.dynamic_model, owned_by="margAI")

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
