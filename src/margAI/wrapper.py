"""The :class:`Wrapper` -- intercept everything around any upstream provider.

A wrapper is the thing you design around, FastAPI-style:

    wrapper = Wrapper(providers={...}, transport=...)

    @wrapper.before("route")        # tag-keyed event (fires for that tag)
    def rewrite(ctx, tag):
        if tag.value:
            ctx.body["model"] = tag.value

    @wrapper.after                  # per-call hook (every call)
    def shape(payload, ctx):
        return payload

Non-streaming calls go through ``before`` -> transport -> ``after``.
Streaming calls go through ``before`` -> transport -> per-chunk ``stream``.
Any failure passes through ``error`` hooks before becoming an OpenAI-shaped
JSON error.

Two layers of extension sit on top of that, and they compose:

- **hooks/events** -- the mechanism. Four phases, per-call or keyed by tag.
- **marglets** -- the unit. One named enhancement (all four phases plus
  metadata) activated per call by a bangtag, and optionally owning the
  routing decision for the calls it is active on.

Request order is: bangtag parse -> marglet/per-call ``before`` -> *route*
(including the dynamic chain) -> transport -> ``after``/``stream`` -> record.
Routing sits after ``before`` on purpose: that is what lets a marglet change
which model serves the call.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from .config import Config
from .core import DONE
from .core.billing import SpendLedger
from .core.configview import ConfigView, ConfigViolation, deep_merge, safe_overlay
from .core.context import SCRATCH_KEY, RequestContext
from .core.errors import ApiError
from .core.events import EventRegistry
from .core.hooks import HookKind, HookRegistry, maybe_await
from .core.intent import RoutingIntent
from .core.marglets import (
    ACTIVE_KEY,
    DYNAMIC_KEY,
    PHASES,
    ROUTE_REASON_KEY,
    Marglet,
    MargletRegistry,
)
from .core.protocol import PreparedRequest, Transport, UpstreamStream
from .core.responses import ControlSignal, ErrorResponse, ImmediateResponse, RouteOverride
from .core.router import ModelRouter, Route
from .core.tags import RESERVED_NAMESPACE, ROOT_PACK, TagRegistry, qualify
from .core.virtual import CostTable, VirtualModels
from .packs import describe as packs_describe
from .packs import install_all
from .providers import build_providers
from .routing import Routing
from .telemetry import CallRecord, Telemetry

#: ``RequestContext.state`` slot for the per-call virtual-model resolver.
CALL_VIRTUALS_KEY = "margAI.call_virtuals"

__all__ = ["GatewayResponse", "StreamHandle", "Wrapper"]

logger = logging.getLogger("margAI.wrapper")

_SSE_DONE = "data: [DONE]\n\n"

# Where the tag-event dispatcher sits among request hooks. Bangtag parsing
# installs at -100, so this lands after it: tags exist by the time the
# tag-keyed handlers (and therefore every marglet) run.
_EVENT_ORDER = -50


class _ShortCircuit(Exception):
    """A ``before`` handler answered the call itself.

    ``ctx.respond()`` and ``ctx.error()`` both end the call, but they end it
    *inside* :meth:`Wrapper._prepare`, which is where the request hooks run and
    where the provider is still unresolved. Raising keeps that decision on the
    way out rather than threading a second return type through every caller.
    The message is never read -- the signal carries the readable body -- so it
    names the type rather than repeating the payload.
    """

    def __init__(self, signal: ImmediateResponse | ErrorResponse, ctx: RequestContext) -> None:
        super().__init__(type(signal).__name__)
        self.signal = signal
        # Carried rather than read back off the exception: the caller still has
        # no context of its own, because `_prepare` never returned one. Without
        # this the answered call could not be recorded, and a cache that leaves
        # no trace cannot be measured.
        self.ctx = ctx


@dataclass
class GatewayResponse:
    body: dict[str, Any]
    status: int = 200


class Wrapper:
    def __init__(
        self,
        *,
        providers: dict[str, Any],
        transport: Transport,
        prefix: str = "margAI",
        expose: str = "prefixed",
        default_provider: str | None = None,
        telemetry: Telemetry | None = None,
        config: Config | None = None,
        name: str = "margAI",
        dynamic_model: str = "dynamic",
    ) -> None:
        self.name = name
        # Which pack registrations belong to, and which namespace they default
        # to. Both are set by the `pack()` context manager; outside one, this is
        # the application's own code. The application's namespace follows the
        # model `prefix`, so a deployment that calls itself "veda" writes
        # `!veda: terse` rather than `!margAI: terse`. The framework's built-in
        # pack still registers into `margAI` explicitly, so both directives
        # work side by side (see builtin_tags.install).
        self._pack: str = ROOT_PACK
        self._namespace: str = prefix or RESERVED_NAMESPACE
        self.providers = providers
        self.transport = transport
        self.config = config
        self._hooks = HookRegistry()
        self._events = EventRegistry(self._namespace)
        self.marglets = MargletRegistry()
        self.router = ModelRouter(
            providers,
            prefix=prefix,
            expose=expose,
            default_provider=default_provider,
            dynamic_model=dynamic_model,
        )
        self.routing = Routing(self.router, default_provider=default_provider)
        # Estimated spend per model, fed by `_record` on the way out of every
        # call. The budget-based and usage-based strategies read it; nothing
        # else does, and it never leaves this process.
        self.ledger = SpendLedger(
            config.billing if config else None,
            priced=bool(config.telemetry.costs) if config else False,
        )
        self.virtuals = VirtualModels(
            prefix,
            (config.models or {}).values() if config else (),
            CostTable(config.telemetry.costs) if config else CostTable(),
            self.ledger,
        )
        # Fail at construction, not on the first request that happens to use
        # the ambiguous id. See VirtualModels.check_collisions.
        self.virtuals.check_collisions(providers.keys(), self.router.model_ids())
        # Kept for per-call overlays, which rebuild a resolver from the
        # deployment's policies plus the caller's.
        self._virtual_costs = self.virtuals.costs
        # The deployment's own per-pack config, frozen once at construction.
        # Per-call overlays merge onto this rather than rebuilding it.
        # Built from the policies themselves, not from `describe()`'s output:
        # a report entry carries an `id` the strict `[models.*]` parser rejects,
        # and a base table that cannot be re-parsed cannot be merged with a
        # caller's override -- the merge would come back empty and the caller's
        # `[models.*]` would silently do nothing.
        models = config.models if config else None
        self._base_config = ConfigView(
            packs=dict((config.packs if config else None) or {}),
            models={
                # `name` dropped: it is the table key, and re-adding it makes
                # the table unparseable by the very parser that accepts a
                # caller's `[models.*]`.
                name: {k: v for k, v in dataclasses.asdict(policy).items() if k != "name"}
                for name, policy in (models or {}).items()
            },
            source="deployment",
        )
        self.telemetry = telemetry or Telemetry(_disabled_telemetry_config())
        self._catalog_cache: list[dict[str, Any]] | None = None
        self._catalog_at = 0.0
        self._catalog_ttl = config.gateway.models_cache_ttl if config else 300.0
        # Wired here, not lazily on first use: registration order must not
        # decide where the event dispatchers sit relative to user hooks.
        self._wire_event_dispatch()

        # The built-in tag pack ships inside this distribution, so the
        # `margAI.packs` entry point group -- which only ever finds *third-party*
        # packs -- cannot report it. Installing it through the same machinery
        # keeps one code path, gives it a record for `margAI doctor`, and means
        # `route=` / `provider=` / `model=` work with no setup. Opt out by name:
        #   [packs."margAI.builtin_tags"]
        #   enabled = false

        # Third-party packs, found through the `margAI.packs` entry point
        # group.
        self.pack_failures: list[Any] = []
        self.packs = install_all(
            self,
            only=(config.gateway.packs_only if config else None),
            enabled=bool(config.gateway.pack_discovery if config else True),
        )
        self.pack_failures = [r.failure for r in self.packs if r.failure]

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: Config,
        transport: Transport | None = None,
        *,
        load_hooks: bool = True,
        name: str = "margAI",
    ) -> Wrapper:
        from .transport.httpx import HttpxTransport

        transport = transport or HttpxTransport(timeout=config.gateway.timeout)
        wrapper = cls(
            providers=build_providers(config),
            transport=transport,
            prefix=config.gateway.prefix,
            expose=config.gateway.expose,
            default_provider=config.gateway.default_provider,
            telemetry=Telemetry(config.telemetry, label=config.gateway.prefix),
            config=config,
            name=name,
            dynamic_model=config.gateway.dynamic_model,
        )
        if load_hooks:
            for entry in config.hooks.load:
                _load_hook_module(entry, wrapper)
        return wrapper

    async def aclose(self) -> None:
        closer = getattr(self.transport, "aclose", None)
        if closer is not None:
            await closer()
        self.telemetry.close()

    # -- event decorators ----------------------------------------------------
    #
    # FastAPI-style surface. Each decorator has two modes selected by whether
    # the first positional arg is a string (or `tag=` is given):
    #
    #   @app.before("route")            # tag-keyed event: fn(ctx, tag)
    #   @app.before                     # per-call hook: fn(ctx)
    #
    # Tag-keyed events are stored on the wrapper's EventRegistry (see
    # `app.events`) and dispatched automatically at the matching phase for
    # every call carrying that tag (see `ctx.tags`). `name=`,
    # when given without a tag, is the per-call hook's display name.

    def on(
        self,
        name: str,
        *,
        phase: str = "before",
        order: int = 0,
        replace: bool = False,
        namespace: str | None = None,
    ) -> Any:
        """Register a handler for a bangtag. The uniform tag entry point.

        ``@app.on("think")`` is the request phase, which is what a tag almost
        always wants; ``phase="stream"`` puts it in the per-chunk path and so
        on. Equivalent to ``@app.<phase>("name")``, spelled uniformly so a pack
        can register across phases without four different decorators.

            @app.on("supply-chain", namespace="sc")
            def supply_chain(ctx, tag):
                ctx.add_system_prompt(AUDIT_INSTRUCTIONS)

        The decorated function is returned unchanged, so it stays an ordinary
        callable you can invoke directly.

        Handlers accumulate: two claims on one tag both run, in ``(order,
        registration)`` order. That is how ``!margAI: think`` can steer routing
        *and* set ``reasoning_effort`` without either handler knowing about the
        other. Displacing another pack's handler is a different thing entirely
        and is refused unless you are writing your own application's code --
        see :meth:`pack`.
        """
        return self._events.on(
            name,
            phase=phase,
            order=order,
            replace=replace,
            namespace=namespace or self._namespace,
            pack=self._pack,
        )

    @contextmanager
    def pack(self, name: str, *, namespace: str | None = None, doc: str = "") -> Any:
        """Group the handlers registered inside this block as one pack.

        The pack owns its namespace, and once claimed no other pack may
        register into it -- so two domain packs can both define ``approve``
        without ever meeting, and the directive says who owns what.

            def install(app):
                with app.pack("supply-chain", namespace="sc", doc="Supply chain"):
                    @app.on("audit")
                    def audit(ctx, tag): ...

        The pack name is recorded on every handler for ``describe()`` and for
        the conflict messages. Registering outside any block is the
        application's own code, recorded as :data:`~margAI.core.tags.ROOT_PACK`.
        """
        ns = namespace or name
        self._events.tags.reserve(ns, name)
        previous = (self._pack, self._namespace)
        self._pack, self._namespace = name, ns
        try:
            yield self
        finally:
            self._pack, self._namespace = previous

    @property
    def tags(self) -> TagRegistry:
        """The tag registry: every handler, its phase, its pack, its doc.

        Read it to answer "what can this gateway's bangtags actually do?" --
        the same source the unknown-tag error and ``margAI doctor`` use, so it
        cannot drift from what is really registered.
        """
        return self._events.tags

    @property
    def namespace(self) -> str:
        """The application's own directive namespace, as configured.

        Defaults to the model ``prefix``, so ``prefix = "veda"`` means
        application handlers fire on ``!veda: <tag>``. Handlers registered
        inside ``app.pack(...)`` belong to that pack's namespace instead.

        This is the value ``install_bangtags(app)`` parses by default, so the
        two never disagree about which directive spelling activates a handler.
        """
        return self._namespace

    def before(
        self, fn: Any = None, *, tag: Any = None, name: Any = None, order: int = 0, replace: bool = False
    ) -> Any:
        """Request phase.

        - ``@app.before("name")`` / ``@app.before(tag="name")``: an event
          handler ``fn(ctx, tag)`` run when the call's tags include ``name``.
        - ``@app.before``: a per-call hook ``fn(ctx)`` on every request.
        """
        if tag is not None or isinstance(fn, str):
            if tag is None:
                tag = fn
            return self._events.before(
                tag, order=order, replace=replace, namespace=self._namespace, pack=self._pack
            )
        return self._decorate(HookKind.REQUEST, fn, name, order)

    def after(self, fn: Any = None, *, tag: Any = None, name: Any = None, order: int = 0, replace: bool = False) -> Any:
        """Response phase (non-streaming).

        - ``@app.after("name")``: event handler ``fn(payload, ctx, tag) ->
          payload``.
        - ``@app.after``: per-call hook ``fn(payload, ctx) -> payload``.
        """
        if tag is not None or isinstance(fn, str):
            if tag is None:
                tag = fn
            return self._events.after(
                tag, order=order, replace=replace, namespace=self._namespace, pack=self._pack
            )
        return self._decorate(HookKind.RESPONSE, fn, name, order)

    def stream(
        self, fn: Any = None, *, tag: Any = None, name: Any = None, order: int = 0, replace: bool = False
    ) -> Any:
        """Stream phase (per SSE chunk).

        - ``@app.stream("name")``: event handler ``fn(chunk, ctx, tag,
          scratch) -> chunk | None`` (``None`` drops the chunk).
        - ``@app.stream``: per-call hook ``fn(chunk, ctx) -> chunk | None``.
        """
        if tag is not None or isinstance(fn, str):
            if tag is None:
                tag = fn
            return self._events.stream(
                tag, order=order, replace=replace, namespace=self._namespace, pack=self._pack
            )
        return self._decorate(HookKind.STREAM, fn, name, order)

    def error(self, fn: Any = None, *, tag: Any = None, name: Any = None, order: int = 0, replace: bool = False) -> Any:
        """Failure phase.

        - ``@app.error("name")``: event handler ``fn(exc, ctx, tag) ->
          dict | None`` (a dict takes over the error body).
        - ``@app.error``: per-call hook ``fn(exc, ctx) -> dict | None``.
        """
        if tag is not None or isinstance(fn, str):
            if tag is None:
                tag = fn
            return self._events.error(
                tag, order=order, replace=replace, namespace=self._namespace, pack=self._pack
            )
        return self._decorate(HookKind.ERROR, fn, name, order)

    @property
    def events(self) -> EventRegistry:
        """The tag-keyed event registry backing the ``before/after/stream/
        error`` decorators. Read it for ``all()`` / ``describe()``."""
        return self._events

    def add_handler(self, handler: Any) -> Any:
        """Register an :class:`~margam.EventHandler` group (or any object
        whose methods are named ``{event}_{tag}``) and return it. Equivalent
        to decorating each method with the matching ``@app.<event>(tag)``."""
        self._events.add(handler)
        return handler

    def _wire_event_dispatch(self) -> None:
        """Wire the tag-event dispatchers into the hook pipeline (once, from
        ``__init__``, so registration order cannot move them).

        ``_dispatch_before`` records the marglets the call activated *before*
        running any handler, which is what populates ``ctx.marglets`` for every
        later phase -- including ``error``, where it is the only way a handler
        can tell what the call asked for. The other three rely on it having
        run; they cannot record it themselves, because an earlier request hook
        may have raised first.
        """

        @self.before(name="events:before", order=_EVENT_ORDER)
        async def _dispatch_before(ctx: RequestContext) -> Any:
            tags = self._dispatch_tags(ctx)
            if not tags:
                return None
            self._check_tags(tags)
            if self.marglets:
                ctx.state[ACTIVE_KEY] = [m.name for m in self.marglets.active(tags)]
            # Returned, not just awaited: a handler may have taken the call over
            # with `ctx.route_to`/`ctx.respond`/`ctx.error`, and `_prepare` is
            # the only thing that can act on that.
            return await self._events.run_before(tags, ctx)

        @self.after(name="events:after")
        async def _dispatch_after(payload: dict[str, Any], ctx: RequestContext) -> dict[str, Any]:
            tags = self._dispatch_tags(ctx)
            if not tags:
                return payload
            return await self._events.run_after(tags, payload, ctx)

        @self.stream(name="events:stream")
        async def _dispatch_stream(chunk: dict[str, Any], ctx: RequestContext) -> dict[str, Any] | None:
            tags = self._dispatch_tags(ctx)
            if not tags:
                return chunk
            scratch = ctx.state.setdefault(SCRATCH_KEY, {})
            return await self._events.run_stream(tags, chunk, ctx, scratch)

        @self.error(name="events:error")
        async def _dispatch_error(exc: Exception, ctx: RequestContext) -> dict[str, Any] | None:
            tags = self._dispatch_tags(ctx)
            if not tags:
                return None
            return await self._events.run_error(tags, exc, ctx)

    def _check_tags(self, tags: list[Any]) -> None:
        """Apply ``gateway.on_unknown_tag`` to tags no handler claims.

        A bangtag that silently does nothing is the worst failure mode of a tag
        system: the caller believes they asked for something. Whether a
        *miss* is an error is not something the framework can decide -- an
        unrecognised tag is a typo on one deployment and a pack that simply is
        not installed here on another -- so the registry reports the miss and
        ``on_unknown_tag`` decides what it means.

        Claims are counted across every phase, so a tag handled only in ``after``
        is known, and a namespaced tag is compared qualified, so ``!hr: approve``
        is not reported as an unknown ``approve``.
        """
        known = self.tags.known()
        unknown = [t for t in tags if qualify(t) and qualify(t) not in known]
        if not unknown:
            return
        detail = ", ".join(sorted({qualify(t) for t in unknown}))
        policy = (self.config.gateway.on_unknown_tag if self.config else "warn") or "warn"
        if policy == "ignore":
            return
        if policy == "error":
            installed = ", ".join(sorted(self.tags.namespaces())) or "none"
            raise ApiError(
                400,
                f"unknown bangtag(s): {detail}. Installed directives: {installed}",
                error_type="invalid_request_error",
                param="bangtags",
            )
        if policy != "warn":
            logger.warning("gateway.on_unknown_tag=%r is not a known policy; treating it as 'warn'", policy)
        for qualified in sorted({qualify(t) for t in unknown}):
            logger.warning(
                "no handler claims bangtag %r -- it will be ignored. "
                "Register one with @app.on(%r), or set gateway.on_unknown_tag.",
                qualified,
                qualified.rpartition(":")[2],
            )

    def _dispatch_tags(self, ctx: RequestContext) -> list[Any]:
        """The tags to dispatch this phase, in *handler* order.

        Tag-keyed events run in the order the tags were written, which for
        marglets would make the user decide execution order by reordering
        their bangtag. Marglets declare an ``order`` and the registry resolves
        ties by name, so that is what wins here: marglet tags first, in
        registry order, then any plain event tags in their original order.

        With no marglets registered this is the identity, so tag-keyed events
        behave exactly as they did before marglets existed.
        """
        tags = ctx.tags
        if not tags or not len(self.marglets):
            return tags
        rank = {m.name: i for i, m in enumerate(self.marglets.ordered())}
        marglet_tags = [t for t in tags if getattr(t, "name", None) in rank]
        plain_tags = [t for t in tags if getattr(t, "name", None) not in rank]
        return sorted(marglet_tags, key=lambda t: rank[getattr(t, "name", "")]) + plain_tags

    def _decorate(self, kind: Any, fn: Any, name: Any, order: Any) -> Any:
        def deco(func: Any) -> Any:
            self._hooks.register(kind, func, name or getattr(func, "__name__", "hook"), order)
            return func

        return deco if fn is None else deco(fn)

    # -- marglets ------------------------------------------------------------

    def add_marglet(self, marglet: Marglet, *, replace: bool = False) -> Marglet:
        """Register a marglet and wire its phases onto the event registry.

        A marglet's hooks *are* tag-keyed event handlers under the hood, so
        registering one and decorating ``@app.before("name")`` are the same
        mechanism -- which is why the two can never disagree about ordering.

        ``replace=True`` also silences the event registry's replacement
        warning for the phases it rewires: the caller asked for this.
        """
        self.marglets.add(marglet, replace=replace)
        tag = marglet.name
        for phase, hook in marglet.phases.items():
            if hook is None:
                continue
            # The marglet's own `order` goes onto its event handlers, so a
            # marglet registered with several claims on one tag resolves them
            # by the same number it declares for routing. One source of truth.
            getattr(self, phase)(tag=tag, order=marglet.order, replace=replace)(hook)
        if marglet.select is not None:
            # A marglet that can route gets a selector scoped to the calls it
            # is active on, at the front of the chain so an explicit marglet
            # beats a generic policy.
            self.routing.add_selector(
                marglet.select, name=f"marglet:{marglet.name}", order=marglet.order, only_for=[marglet.name]
            )
        return marglet

    def marglet(self, fn: Any = None, *, name: Any = None, order: int = 0, **kwargs: Any) -> Any:
        """Decorator form of :meth:`add_marglet`, with the same two modes as
        the phase decorators::

            @app.marglet("terse", summary="Answer briefly")
            def terse(ctx, tag):
                ctx.add_system_prompt("Answer in as few words as possible.")

            @app.marglet(summary="...")          # name from the function
            def terse(ctx, tag): ...

        Other phases go through ``Marglet(before=..., after=...)`` or
        :meth:`add_marglet_group`.
        """
        if isinstance(fn, str):
            name = name or fn
            fn = None
        if fn is not None:
            self.add_marglet(Marglet(name or fn.__name__, before=fn, order=order, **kwargs))
            return fn

        def deco(func: Any) -> Any:
            self.add_marglet(Marglet(name or func.__name__, before=func, order=order, **kwargs))
            return func

        return deco

    def add_marglet_group(self, group: Any) -> Any:
        """Register a group of marglets from ``{name}_{phase}``-style methods.

        The marglet-level analogue of :meth:`add_handler`: one object holding
        several related marglets, e.g.::

            class Bullets:
                def terse_before(self, ctx, tag): ...
                def terse_after(self, payload, ctx, tag): ...
                def bullets_before(self, ctx, tag): ...

            app.add_marglet_group(Bullets())

        The name is the part *before* the phase, matching ``add_handler``'s
        ``{event}_{name}`` shape transposed into ``{name}_{phase}``.
        """
        found: dict[str, dict[str, Any]] = {}
        for attr in dir(group):
            if attr.startswith("_") or "_" not in attr:
                continue
            name, _, phase = attr.rpartition("_")
            if not name or phase not in PHASES:
                continue
            fn = getattr(group, attr)
            if callable(fn):
                found.setdefault(name, {})[phase] = fn
        for marglet_name, phases in found.items():
            self.add_marglet(Marglet(marglet_name, **phases))
        return group

    # -- request pipeline ---------------------------------------------------

    # The routed surface. Every kind here is backed by a real Provider method
    # and covered by tests; anything absent is not routable. See
    # docs/ENDPOINTS.md before adding one.
    _KIND_PREPARE: ClassVar[dict[str, str]] = {
        "chat": "prepare_chat",
        "completions": "prepare_completions",
        "embeddings": "prepare_embeddings",
        "images_generations": "prepare_images_generations",
        "audio_transcriptions": "prepare_audio_transcriptions",
    }

    _KIND_PARSE: ClassVar[dict[str, str]] = {
        "chat": "parse_response",
        "completions": "parse_completion_response",
        "embeddings": "parse_embeddings_response",
        "images_generations": "parse_images_generations_response",
        "audio_transcriptions": "parse_audio_transcriptions_response",
    }

    _KIND_PARSE_CHUNK: ClassVar[dict[str, str]] = {
        "chat": "parse_chunk",
        "completions": "parse_completion_chunk",
    }

    _STREAMABLE_KINDS: ClassVar[set[str]] = {"chat", "completions"}

    @staticmethod
    def _provider_method(provider: Any, provider_name: str, method: str, kind: str) -> Any:
        """Look a kind's provider method up, degrading to 404 if it is absent.

        A kind in ``_KIND_PREPARE`` that a given provider has not implemented
        is a 404 ("this provider does not speak that"), not an
        ``AttributeError`` that surfaces as a redacted 500. The base
        :class:`~margAI.providers.Provider` defines these fallbacks, so this is
        a backstop for a kind added to the tables without them.
        """
        fn = getattr(provider, method, None)
        if fn is None:
            raise ApiError(
                404,
                f"Provider '{provider_name}' does not support {kind}",
                error_type="invalid_request_error",
                param="model",
            )
        return fn

    def _call_config(self, overlay: str | Mapping[str, Any] | None) -> ConfigView:
        """Merge a caller's per-call config over the deployment's own.

        The base comes from ``[packs.*]`` and ``[models.*]`` in the config
        file, so an operator sets a default and a caller overrides one key --
        and a caller's override is checked against the same whitelist whether
        it arrived as TOML from a HTTP body or as a dict from Python.
        """
        if overlay is None:
            return self._base_config
        incoming = safe_overlay(overlay)
        if not incoming.packs and not incoming.models:
            return self._base_config
        # Deep merge, not `{**a, **b}`. An operator sets `[packs.fast]
        # endpoint = ...` and a caller overrides one field of it; a top-level
        # merge would drop the rest of the table and the caller's own value
        # would be the only one left standing.
        return ConfigView(
            packs=deep_merge(self._base_config.packs, incoming.packs),
            models=deep_merge(self._base_config.models, incoming.models),
            source="call",
        )

    async def _prepare(
        self, body: dict[str, Any], stream: bool, kind: str = "chat", config: Any = None
    ) -> tuple[RequestContext, Any, PreparedRequest]:
        """Run request hooks, resolve the route, and build the upstream request.

        The body is shallow-copied, not deep-copied: a chat body is a list of
        messages that can be tens of kilobytes, and ``deepcopy`` on every
        request showed up as real latency. Hooks get the top-level dict to
        replace (``return {...}``) and mutate ``ctx.body`` in place; the
        nested message list is shared with the caller's parsed JSON, which is
        exactly what the pre-existing in-place hook contract already assumed.
        """
        ctx = RequestContext(body=dict(body), stream=stream, gateway=self.config)
        # Resolved before any handler runs, so a `before` pack is the first
        # thing able to see it and a `stream` pack sees the same view it did.
        try:
            ctx.config = self._call_config(config)
        except ConfigViolation as exc:
            # A caller sending config is a client, so a refusal is their 400,
            # not an internal error -- and it happens before any handler runs,
            # so a rejected config cannot have had a side effect.
            raise ApiError(
                400, str(exc), error_type="invalid_request_error", param="config"
            ) from exc
        try:
            override: RouteOverride | None = None
            for hook in self._hooks.requests():
                out = await maybe_await(hook.fn(ctx))
                if out is None:
                    continue
                if isinstance(out, ControlSignal):
                    # A `before` handler answered the call itself. A route
                    # override is applied and the pipeline carries on; a
                    # response or an error ends it here, before any provider is
                    # resolved -- so a call answered from a cache does not need
                    # a model that exists, and a rejected one is never attempted.
                    if isinstance(out, RouteOverride):
                        override = out
                        break
                    if isinstance(out, (ImmediateResponse, ErrorResponse)):
                        raise _ShortCircuit(out, ctx)
                    # A fourth signal type, added later without a case here.
                    # Failing loudly beats treating a control signal as a
                    # rewritten request body.
                    raise TypeError(f"unhandled control signal: {type(out).__name__}")
                ctx.body = out

            if override is not None:
                # Recorded before it is overwritten, so telemetry still shows
                # what the caller asked for next to what it was given.
                requested = ctx.body.get("model")
                ctx.body["model"] = override.model
            else:
                requested = None

            route = self._resolve(ctx)
            if override is not None:
                # `_resolve` only stamps a reason on a dynamic call, so a
                # concrete override would otherwise be unattributable in
                # telemetry.
                ctx.state[ROUTE_REASON_KEY] = "handler:route"
            ctx.request_model = ctx.body.get("model") if requested is None else requested
            ctx.provider, ctx.upstream_model = route.provider, route.model
            ctx.body["model"] = route.model
            provider = self.providers[route.provider]

            prepare_method = self._KIND_PREPARE.get(kind)
            if prepare_method is None:
                raise ApiError(400, f"Unknown endpoint kind: {kind}", error_type="invalid_request_error", param="model")
            prepared = self._provider_method(provider, route.provider, prepare_method, kind)(ctx)
        except BaseException as exc:
            # Hand the context to the error so hooks and telemetry can report
            # the call that failed, not just that something did. This covers
            # ordinary exceptions too: a before-hook that raises is the most
            # common failure of all, and it must still reach the error phase.
            with suppress(Exception):  # exotic exception types may forbid attrs
                exc.ctx = ctx  # type: ignore[attr-defined]
            raise
        return ctx, provider, prepared

    def _resolve(self, ctx: RequestContext) -> Route:
        """Route the call. Runs *after* every ``before`` hook, which is what
        lets a marglet steer routing.

        Three shapes of ``model`` field, in the order they are tried:

        1. a virtual model id (``margAI/fast``) -- resolved against the live
           catalogue by its policy;
        2. ``margAI/dynamic`` (or the bare ``dynamic`` when nothing claims it)
           -- handed to the selector chain;
        3. anything else -- a plain name lookup.
        """
        requested = ctx.body.get("model") or ""
        if self.virtuals.is_virtual(requested):
            return self._route(ctx, self._resolve_virtual(ctx, requested))
        if not self.router.is_dynamic(requested):
            return self.router.resolve(requested)

        ctx.state[DYNAMIC_KEY] = True
        if (pinned := self._pinned_route(ctx)) is not None:
            return self._route(ctx, pinned)
        return self._route(ctx, self.routing.select(intent=ctx.intent, ctx=ctx))

    def _resolve_virtual(self, ctx: RequestContext, requested: str) -> Route:
        """Resolve a virtual model id, honouring the call's own constraints.

        The caller's ``not=``/``only=``/``route=`` constraints are applied
        *before* the policy, so the policy can only narrow the field. That
        ordering matters for the stateful strategies: a round-robin cursor
        would otherwise hand out slots to targets this call excluded.
        """
        ctx.state[DYNAMIC_KEY] = True
        if (pinned := self._pinned_route(ctx)) is not None:
            return pinned
        pool = ctx.intent.filter(self.router.candidates())
        route = self._virtuals_for(ctx).resolve(requested, ctx.intent, pool)
        assert route is not None  # is_virtual() was true one line up
        return route

    def _virtuals_for(self, ctx: RequestContext) -> VirtualModels:
        """The virtual models this call routes under.

        A caller's ``[models.<name>]`` overlay gets a real resolver rather than
        only being readable by pack authors. Allowing the table in the
        whitelist and then ignoring it at routing time would be the worst of
        both worlds: the config is accepted, validated, and silently does
        nothing.

        The per-call set is built once per call and cached, because a table
        rebuild is pure overhead on a path that runs per request.
        """
        cached = ctx.state.get(CALL_VIRTUALS_KEY)
        if cached is not None:
            return cached
        base = self.virtuals
        # Equal, not merely non-empty: a caller's overlay that restates the
        # deployment's models must not rebuild the resolver, because a fresh
        # `VirtualModels` means a fresh round-robin cursor, and every call
        # would then be handed the same first target.
        if ctx.config.models == self._base_config.models:
            ctx.state[CALL_VIRTUALS_KEY] = base
            return base
        from .config import _parse_models

        merged = base.policies()
        # `_parse_models` takes the config root, not the `[models]` table.
        merged.update(_parse_models({"models": dict(ctx.config.models)}))
        override = VirtualModels(
            base.prefix,
            merged.values(),
            self._virtual_costs,
        )
        ctx.state[CALL_VIRTUALS_KEY] = override
        return override

    def _pinned_route(self, ctx: RequestContext) -> Route | None:
        """The concrete target a bangtag already named, if it named one.

        "!margAI: route=local/qwen" asked for a specific target, which is a
        stronger statement than any policy or selector, so the pin wins.

        It is still subject to the call's own constraints. ``route=`` states a
        preference and ``not=``/``only=`` state policy, and a contradiction
        between the two must fail loudly rather than quietly serve the
        excluded target -- same rule the default_provider fallback follows.
        """
        intent: RoutingIntent = ctx.intent
        if not (intent.provider and intent.model):
            return None
        pinned = self.router.try_resolve(self.router.namespaced_id(intent.provider, intent.model))
        if pinned is None:
            return None
        if not intent.allows(pinned.provider, pinned.model):
            raise ApiError(
                404,
                self._route_error(
                    f"bangtag route={pinned.provider}/{pinned.model} is excluded by this call's constraints"
                ),
                error_type="invalid_request_error",
                param="model",
            )
        return Route(pinned.provider, pinned.model, "bangtag:route")

    @staticmethod
    def _route(ctx: RequestContext, route: Route) -> Route:
        """Record why the dynamic router chose what it chose.

        Every dynamic call gets a reason, so none is unattributable in
        telemetry -- including the ones pinned by a bangtag, which used to
        return early and record nothing.
        """
        ctx.state[ROUTE_REASON_KEY] = route.reason
        return route

    def _route_error(self, detail: str) -> str:
        """A 404 body that says what was asked and what was available."""
        return (
            f"Cannot route {self.router.dynamic_id!r}: {detail}. "
            f"Register a selector with wrapper.routing.add_selector(...) or set gateway.default_provider."
        )

    async def _run_error_hooks(self, exc: Exception, ctx: RequestContext | None) -> dict[str, Any] | None:
        """Run ``error`` hooks for a failure, if there is a context to run
        them against. Prefers the context attached to the error, because the
        local one is still ``None`` when ``_prepare`` itself failed."""
        ctx = getattr(exc, "ctx", None) or ctx
        if ctx is None:
            return None
        return await self._hooks.apply_error(exc, ctx)

    def _default_error_body(self, exc: Exception) -> dict[str, Any]:
        if isinstance(exc, ApiError):
            return exc.body
        # Never echo the exception text to the client; it leaks paths, keys
        # and upstream internals. The detail is in the log, where it belongs.
        logger.error("unhandled error: %r", exc, exc_info=exc)
        return ApiError(500, "Internal server error", error_type="server_error").body

    def _record(
        self,
        ctx: RequestContext,
        *,
        status: int = 200,
        error: str | None = None,
        usage: dict[str, Any] | None = None,
        source: str = "chat",
    ) -> CallRecord:
        tokens: dict[str, Any] = usage or {}
        raw_details = tokens.get("prompt_tokens_details")
        details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
        record = self.telemetry.record(
            source=source,
            request_model=ctx.request_model,
            provider=ctx.provider,
            upstream_model=ctx.upstream_model,
            stream=ctx.stream,
            prompt_tokens=tokens.get("prompt_tokens"),
            completion_tokens=tokens.get("completion_tokens"),
            total_tokens=tokens.get("total_tokens"),
            cached_tokens=details.get("cached_tokens"),
            latency_ms=ctx.elapsed_ms(),
            status=status,
            error=error,
            marglets=tuple(ctx.marglets),
            route_reason=ctx.state.get(ROUTE_REASON_KEY) or "",
        )
        # Feed the spend ledger the same cost telemetry reports, so a
        # budget-based route and a cost report can never disagree. Recorded
        # even when the cost is unknown (an errored or unpriced call): the
        # call still happened, and `least_used` ranks on calls, not dollars.
        self.ledger.record(
            record.provider,
            record.upstream_model,
            self.telemetry.cost_for(record),
        )
        return record

    # -- non-streaming chat / completions -------------------------------------

    async def complete(
        self, body: dict[str, Any], *, kind: str = "chat", config: str | Mapping[str, Any] | None = None
    ) -> GatewayResponse:
        """Run one non-streaming completion.

        ``config`` is an optional per-call configuration -- a TOML string or a
        plain dict -- merged over the deployment's own ``[packs.*]`` config.
        It may only set ``[packs.*]`` and ``[models.*]``; see
        :func:`margAI.core.configview.safe_overlay` for why.
        """
        ctx: RequestContext | None = None
        try:
            ctx, provider, prepared = await self._prepare(body, stream=False, kind=kind, config=config)
            parse_method = self._KIND_PARSE.get(kind)
            if parse_method is None:
                raise ApiError(400, f"Unknown endpoint kind: {kind}", error_type="invalid_request_error", param="model")
            parse = self._provider_method(provider, ctx.provider or "?", parse_method, kind)
            resp = await self.transport.request(prepared)
            payload, status = parse(resp, ctx)
            payload = await self._hooks.apply_response(payload, ctx)
            self.telemetry.emit(self._record(ctx, status=status, usage=provider.extract_usage(payload), source=kind))
            return GatewayResponse(body=payload, status=status)
        except _ShortCircuit as sc:
            return self._answered(sc, kind)
        except ApiError as exc:
            return await self._fail(exc, ctx, kind, exc.status, exc.message)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("unhandled error in wrapper.complete")
            return await self._fail(exc, ctx, kind, 500, repr(exc))

    def _answered(self, sc: _ShortCircuit, kind: str) -> GatewayResponse:
        """Turn a ``before`` handler's answer into a response, and record it.

        The ``after`` and ``error`` phases are skipped: there is no upstream
        answer to shape and no failure to explain. Telemetry still records the
        call, because the caller-facing telemetry is the only place a cache hit
        can be seen at all -- and the record's provider and upstream model are
        empty, which is honest: nothing was resolved. ``route_reason`` says a
        handler answered, so a report can tell that from a served call.
        """
        signal = sc.signal
        if isinstance(signal, ErrorResponse):
            status: int = signal.status
            body = signal.to_dict()
        else:
            status = 200
            body = signal.payload
        sc.ctx.state[ROUTE_REASON_KEY] = "handler:answer"
        self.telemetry.emit(self._record(sc.ctx, status=status, source=kind))
        return GatewayResponse(body=body, status=status)

    async def _fail(
        self, exc: Exception, ctx: RequestContext | None, kind: str, status: int, error: str
    ) -> GatewayResponse:
        """The non-streaming failure path: error hooks, telemetry, then a body."""
        handled = await self._on_failure(exc, ctx, kind=kind, status=status, error=error)
        return GatewayResponse(body=handled if handled is not None else self._default_error_body(exc), status=status)

    async def _on_failure(
        self, exc: Exception, ctx: RequestContext | None, *, kind: str, status: int, error: str
    ) -> dict[str, Any] | None:
        """Run the error hooks and record the failure. Returns the error body
        if a hook took it over.

        The single place a call's failure is both shaped and recorded. The
        non-streaming path turns the return into a response; the streaming
        preflight path re-raises with it, because no SSE byte has been written
        yet and the caller can still answer with JSON. They used to be two
        near-identical bodies that drifted, so a failure could be recorded in
        one path and not the other.
        """
        handled = await self._run_error_hooks(exc, ctx)
        ctx = getattr(exc, "ctx", None) or ctx
        self.telemetry.emit(
            self._record(ctx, status=status, error=error, source=kind)
            if ctx is not None
            else self.telemetry.record(status=status, error=error, source=kind)
        )
        return handled

    # -- streaming chat / completions -----------------------------------------

    async def open_stream(
        self, body: dict[str, Any], *, kind: str = "chat", config: str | Mapping[str, Any] | None = None
    ) -> StreamSource:
        """Open a streaming completion. ``config`` behaves as in :meth:`complete`."""
        """Open the upstream stream. Raises :class:`ApiError` (or any other
        exception) *before* any SSE bytes would be sent, so the caller can
        still respond with a JSON error instead of a broken stream.

        Failures here go through the same ``error`` hooks and telemetry as the
        non-streaming path. They used to skip both, which meant a rejected
        dynamic route or an upstream 401 looked like a call that never
        happened.
        """
        ctx: RequestContext | None = None
        try:
            if kind not in self._STREAMABLE_KINDS:
                # Inside the try, so this rejection is recorded and reaches the
                # error hooks like every other failure.
                raise ApiError(
                    400,
                    f"Kind '{kind}' does not support streaming",
                    error_type="invalid_request_error",
                    param="model",
                )
            ctx, provider, prepared = await self._prepare(body, stream=True, kind=kind, config=config)
            upstream = await self.transport.open_stream(prepared)
            if upstream.status >= 400:
                try:
                    err_body = await upstream.json()
                except Exception:
                    err_body = None
                raise ApiError.from_openai_body(err_body, implicit_status=upstream.status)
        except _ShortCircuit as sc:
            return self._answer_stream(sc, kind)
        except ApiError as exc:
            handled = await self._on_failure(exc, ctx, kind=kind, status=exc.status, error=exc.message)
            if handled is not None:
                # `with_body` returns a new ApiError, so this must be an
                # explicit `raise exc` -- a bare `raise` would re-raise the
                # original and drop the hook's body.
                exc = exc.with_body(handled)  # an error hook took over the body
            raise exc
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("failed to open stream")
            handled = await self._on_failure(exc, ctx, kind=kind, status=500, error=repr(exc))
            if handled is not None:
                # Same contract as the ApiError branch above: a hook that shaped
                # the body must have it survive, or a non-ApiError failure --
                # which is most of them -- silently loses whatever it said.
                raise ApiError(500, "Internal server error", error_type="server_error", body=handled) from exc
            raise
        return StreamHandle(self, ctx, provider, upstream, kind=kind)

    def _answer_stream(self, sc: _ShortCircuit, kind: str) -> StreamSource:
        """Answer a streaming call that a ``before`` handler finished itself.

        ``ctx.error()`` still raises: this is the preflight, no SSE byte has
        been written, and the transport can still answer with a JSON error --
        which is what a client parsing an error expects, and what
        ``_fail`` does everywhere else.

        ``ctx.respond()`` cannot, because the client asked for SSE. The cached
        body goes out as a single frame followed by ``[DONE]``, so a streaming
        client sees a well-formed one-frame stream rather than a JSON body it
        would have to be taught to accept.
        """
        signal = sc.signal
        if isinstance(signal, ErrorResponse):
            self._answered(sc, kind)
            raise ApiError(
                signal.status,
                signal.message,
                error_type=signal.error_type,
                body=signal.to_dict(),
            )
        return _ImmediateStream(signal.payload)

    # -- model catalog ------------------------------------------------------

    @property
    def catalog_ttl(self) -> float:
        return self._catalog_ttl

    async def provider_models(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for name, provider in self.providers.items():
            configured = provider.configured_models()
            if configured:
                out[name] = configured
                continue
            try:
                discovered = await provider.list_models(self.transport)
            except Exception as exc:  # catalog must degrade gracefully
                logger.warning("model discovery failed for '%s': %s", name, exc)
                discovered = []
            out[name] = discovered
        return out

    def describe(self) -> dict[str, Any]:
        """What this gateway can do, as one payload.

        The single source behind ``GET /v1/margAI/tags`` and ``margAI doctor``,
        so the answer a client gets and the answer an operator reads cannot
        drift. Everything here is read from the live registry and router
        rather than from the config that was meant to produce them -- a report
        built from the config would happily describe tags nobody registered.
        """
        registry = self.tags.describe()
        return {
            "gateway": {
                "name": self.name,
                "prefix": self.router.prefix,
                "expose": self.router.expose,
                "dynamic_model": self.router.dynamic_id,
                "default_provider": self.router.default_provider,
                "on_unknown_tag": self.config.gateway.on_unknown_tag if self.config else "warn",
            },
            "tags": {
                namespace: [
                    {
                        "name": entry["name"],
                        # Composed from the *display* namespace, not from
                        # `Tag.qualified` -- that is the folded identity key
                        # used for lookups, and a report printing
                        # `margai:think` when every doc says `margAI:think`
                        # would be a bug in the one place people go to check
                        # what to type.
                        "id": f"{namespace}:{entry['name']}",
                        "phases": entry["phases"],
                        "packs": entry["packs"],
                        "doc": entry["doc"],
                    }
                    for entry in entries
                ]
                for namespace, entries in registry.items()
            },
            "namespaces": sorted(self.tags.namespaces()),
            "virtual_models": self.virtuals.describe(),
            "providers": [
                {
                    "name": name,
                    "kind": getattr(getattr(p, "config", None), "kind", ""),
                    "models": list(p.configured_models()),
                    "key_env": getattr(getattr(p, "config", None), "api_key_env", None),
                }
                for name, p in sorted(self.providers.items())
            ],
            "packs": list(packs_describe(getattr(self, "packs", ()))),
            "config": {
                "source": self.config.source if self.config else None,
                "layers": list(self.config.layers) if self.config else [],
                "per_call_tables": ["packs", "models"],
            },
            "failures": [
                {"name": f.name, "reason": f.reason, "target": f.target}
                for f in getattr(self, "pack_failures", ())
            ],
        }

    async def models(self) -> list[dict[str, Any]]:
        if self._catalog_cache is not None and (time.monotonic() - self._catalog_at) < self.catalog_ttl:
            return self._catalog_cache
        provider_models = await self.provider_models()
        # Always advertise the dynamic id: a client picking from a dropdown
        # (LibreChat) can only use it if it can see it.
        self._catalog_cache = self.router.catalog(provider_models, include_dynamic=True)
        # Virtual models are client-facing ids too. A client picking from a
        # dropdown cannot use "margAI/fast" unless it can see it, and the
        # whole point of the name is that the client does not have to know
        # which real model it will land on.
        known = {item["id"] for item in self._catalog_cache}
        now = int(time.time())
        for entry in self.virtuals.describe():
            if entry["id"] not in known:
                self._catalog_cache.append(
                    {
                        "id": entry["id"],
                        "object": "model",
                        "created": now,
                        "owned_by": "margAI",
                        "parent": None,
                        "virtual": entry,
                    }
                )
        self._catalog_at = time.monotonic()
        return self._catalog_cache

    def invalidate_models_cache(self) -> None:
        """Drop the cached ``/v1/models`` catalog so the next call re-fetches."""
        self._catalog_cache = None
        self._catalog_at = 0.0


class StreamSource(Protocol):
    """What a transport needs from an open stream: the SSE lines.

    Narrower than :class:`StreamHandle` on purpose. A stream can be served
    without ever opening an upstream connection -- a ``before`` handler that
    answered the call from a cache is the real case -- and a transport should
    not have to know which of the two it is holding.
    """

    def lines(self) -> AsyncIterator[str]: ...


class _ImmediateStream:
    """A stream that a ``before`` handler answered without an upstream.

    The payload is one frame: a cached completion is already whole, and
    splitting it into deltas to imitate a live token stream would be inventing
    tokens that were never generated.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    async def lines(self) -> AsyncIterator[str]:
        yield f"data: {json.dumps(self._payload, ensure_ascii=False)}\n\n"
        yield _SSE_DONE


class StreamHandle(StreamSource):
    """Held-open upstream stream + the per-chunk hook pipeline.

    Constructed by :meth:`Wrapper.open_stream`; its :meth:`lines` generator
    is what an SSE transport feeds to the client.
    """

    def __init__(
        self,
        wrapper: Wrapper,
        ctx: RequestContext,
        provider: Any,
        upstream: UpstreamStream,
        *,
        kind: str = "chat",
    ) -> None:
        self._wrapper = wrapper
        self._ctx = ctx
        self._provider = provider
        self._upstream = upstream
        self._kind = kind
        parse_chunk_method = wrapper._KIND_PARSE_CHUNK.get(kind)
        if parse_chunk_method is None:
            raise ApiError(
                400,
                f"Kind '{kind}' does not support streaming",
                error_type="invalid_request_error",
                param="model",
            )
        self._parse_chunk = wrapper._provider_method(provider, provider.name, parse_chunk_method, kind)
        self._last_usage: dict[str, Any] | None = None

    def lines(self) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            done = False
            error: str | None = None
            status = 200
            try:
                async for raw in self._upstream.lines():
                    chunk = self._parse_chunk(raw, self._ctx)
                    if chunk is None:
                        continue
                    if chunk is DONE:
                        yield _SSE_DONE
                        done = True
                        break
                    out = await self._wrapper._hooks.apply_stream(chunk, self._ctx)
                    if out is None:
                        continue
                    if isinstance(out, dict):
                        usage = self._provider.extract_usage(out)
                        if usage:
                            self._last_usage = usage
                    yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
                if not done:
                    yield _SSE_DONE
            except ApiError as exc:
                status = exc.status
                error = str(exc)
                body = (await self._wrapper._run_error_hooks(exc, self._ctx)) or exc.body
                yield f"data: {json.dumps(body, ensure_ascii=False)}\n\n"
                yield _SSE_DONE
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("stream failed")
                status = 500
                error = repr(exc)
                err = ApiError(500, f"Stream failed: {exc}", error_type="server_error")
                body = (await self._wrapper._run_error_hooks(exc, self._ctx)) or err.body
                yield f"data: {json.dumps(body, ensure_ascii=False)}\n\n"
                yield _SSE_DONE
            finally:
                self._wrapper.telemetry.emit(
                    self._wrapper._record(
                        self._ctx, status=status, error=error, usage=self._last_usage, source=self._kind
                    )
                )

        return _gen()


def _load_hook_module(entry: str, wrapper: Wrapper) -> None:
    module_name, sep, attr = entry.partition(":")
    module = importlib.import_module(module_name)
    # Either a module (`register` attribute) or a resolved dotted path; both
    # are dynamic, so the callable is only known at runtime.
    register: Any = module
    if sep and attr:
        for part in attr.split("."):
            register = getattr(register, part)
    else:
        register = getattr(module, "register", None)
        if register is None:
            raise ValueError(f"hook module '{entry}' has no register(wrapper) function or 'module:attr' form")
    result = register(wrapper)
    if isinstance(result, Wrapper) and result is not wrapper:
        raise ValueError(
            f"hook '{entry}' returned a different wrapper; pass the same wrapper instance or register in place"
        )


def _disabled_telemetry_config() -> Any:
    from .config import TelemetryConfig

    return TelemetryConfig(enabled=False, emit="none")