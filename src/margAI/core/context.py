"""Per-call context passed to interceptors."""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .configview import EMPTY_CONFIG, ConfigView
from .intent import INTENT_KEY, RoutingIntent
from .marglets import ACTIVE_KEY, DYNAMIC_KEY
from .responses import ErrorResponse, ImmediateResponse, RouteOverride

__all__ = ["SCRATCH_KEY", "STATE_KEY", "RequestContext"]

# Keys inside ``state`` that the framework itself reads. They live here rather
# than in ``events`` because the context is the thing they index into, and
# ``events`` already imports this module transitively.
#
# ``STATE_KEY`` is where the tag-keyed event dispatch reads a call's tags from --
# ``margAI.bangtag.install_bangtags`` writes there, and a handler group that
# prefers its own tag source can write anywhere and set ``tags_key``.
STATE_KEY = "bangtags"
# Per-call mutable scratch space shared by ``stream`` event handlers.
SCRATCH_KEY = "events.scratch"


@dataclass(slots=True)
class RequestContext:
    """Mutable, per-request state carried through the wrapper pipeline.

    - ``body``: the OpenAI-shaped request body (``messages``, ``model``,
      sampling params...). Hooks may mutate it in place or return a
      replacement dict from a ``request`` hook.
    - ``state``: arbitrary scratch space private to one call. Use this to
      thread data from a ``request`` hook to a ``response``/``stream`` hook
      (this is the ``ctx.state`` pattern from the original margAI sketch).
    - ``provider`` / ``upstream_model`` / ``request_model``: populated by the
      router before transport is reached.
    """

    body: dict[str, Any]
    stream: bool = False
    provider: str | None = None
    upstream_model: str | None = None
    request_model: str | None = None
    state: dict[str, Any] = field(default_factory=dict)
    #: Per-call configuration for this request. Defaults to the empty view so
    #: a handler that asks for configuration it was not given gets its own
    #: defaults rather than an AttributeError -- adding a pack must never mean
    #: every existing handler has to become config-aware.
    config: ConfigView = EMPTY_CONFIG
    #: The gateway's own configuration. Separate from :attr:`config` because
    #: "how this call is tuned" and "how this deployment is set up" are
    #: different questions, and a pack that can only see the second one cannot
    #: be given per-call behaviour.
    gateway: Any = None
    started_at: float = field(default_factory=time.perf_counter)
    #: Where :attr:`tags` reads from. Defaults to :data:`STATE_KEY`; set it
    #: alongside ``set_tags(..., key=...)`` when your tag source is elsewhere.
    tags_key: str = STATE_KEY

    # -- tags and marglets --------------------------------------------------

    @property
    def tags(self) -> list[Any]:
        """The bangtags carried by this call (empty when the layer is off)."""
        return self.state.get(self.tags_key) or []

    def set_tags(
        self,
        tags: Iterable[Any],
        *,
        key: str | None = None,
        namespace: str | None = None,
    ) -> None:
        """Record this call's tags, and read them back through :attr:`tags`.

        The one-line way to feed the tag-keyed events from your own request
        hook -- the alternative to ``install_bangtags``. Pass ``key`` to store
        somewhere other than :data:`STATE_KEY`; :attr:`tags` follows it.

        Pass ``namespace`` to claim the tags *for one namespace*, replacing only
        that namespace's previous contribution and leaving every other intact.
        This is what lets several ``install_bangtags`` hooks share one tag list:
        they each run in their own request hook, they each parse only their own
        directive, and without this the last one to run would silently discard
        the others -- leaving ``!margAI:`` working and ``!sc:`` invisible, or
        worse, the reverse, depending on registration order.

        A consequence worth knowing: because each hook only ever sees its own
        directive, the order tags end up in is the order the parsers ran, not
        the order they were written. Do not rely on the order of tags from
        *different* namespaces. Order *within* one namespace is the written
        order, and ordering that is actually load-bearing is a handler's
        ``order``, not a tag's position.

        Order matters: the tag-keyed events dispatch at a fixed ``order=-50``,
        so a hook that sets tags has to run before that (``order=-100``, the
        same slot ``install_bangtags`` uses). A hook at the default ``order=0``
        runs *after* the dispatch and its tags arrive too late.

            @app.before(order=-100)
            def tag_it(ctx):
                ctx.set_tags(my_tags_for(ctx))
        """
        if key is not None:
            self.tags_key = key
        incoming = list(tags)
        if namespace is None:
            self.state[self.tags_key] = incoming
            return
        from .tags import namespace_key

        key_ns = namespace_key(namespace)
        kept = [
            tag
            for tag in self.state.get(self.tags_key) or []
            if namespace_key(getattr(tag, "namespace", "") or "") != key_ns
        ]
        self.state[self.tags_key] = [*kept, *incoming]

    @property
    def marglets(self) -> list[str]:
        """Names of the marglets this call activated, in dispatch order."""
        return list(self.state.get(ACTIVE_KEY) or [])

    @property
    def intent(self) -> RoutingIntent:
        """This call's routing intent, created on first access from its tags.

        The intent is immutable, so a marglet refines it with
        :meth:`steer` (``ctx.steer(provider="local")``) or by assigning
        ``ctx.intent = ctx.intent.with_(...)``. The dynamic router reads
        whatever is here by the time routing happens.
        """
        intent = self.state.get(INTENT_KEY)
        if intent is None:
            intent = RoutingIntent.from_tags(self.tags)
            self.state[INTENT_KEY] = intent
        return intent

    @intent.setter
    def intent(self, value: RoutingIntent) -> None:
        self.state[INTENT_KEY] = value

    def steer(self, **changes: Any) -> RoutingIntent:
        """Refine the intent and return it: ``ctx.steer(provider="local")``."""
        self.intent = self.intent.with_(**changes)
        return self.intent

    @property
    def dynamic(self) -> bool:
        """True when the client asked for the reserved dynamic model id."""
        return bool(self.state.get(DYNAMIC_KEY, False))


    # -- body helpers -------------------------------------------------------

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.body.get("messages", [])

    @property
    def prompt(self) -> Any:
        """The legacy-completions prompt (``/v1/completions``); None on chat calls."""
        return self.body.get("prompt")

    def add_system_prompt(self, text: str, *, prepend: bool = True) -> None:
        """Insert or merge a system message.

        If a system message already exists its content is merged with
        ``text`` (prepended by default, since stable instructions belong
        early for prefix caching).
        """
        msgs = self.body.setdefault("messages", [])
        for msg in msgs:
            if msg.get("role") == "system":
                existing = str(msg.get("content", ""))
                msg["content"] = (
                    (text + "\n\n" + existing) if prepend else (existing + "\n\n" + text)
                )
                return
        msgs.insert(0, {"role": "system", "content": text})

    def last_user_message(self) -> str | None:
        for msg in reversed(self.messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"]
        return None

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started_at) * 1000)

    # -- Gateway API helpers ------------------------------------------------
    # These methods provide the simplified Gateway API for tag functions

    def user_message(self) -> str:
        """Get the last user message content.
        
        Returns:
            The last user message text, or empty string if none exists
        
        Example::
        
            @app.tag("analyze")
            def analyze(ctx, tag):
                msg = ctx.user_message()
                if len(msg) > 1000:
                    ctx.add_system("This is a long query, be thorough.")
        """
        return self.last_user_message() or ""
    
    def user_message_length(self) -> int:
        """Get word count of the last user message.
        
        Returns:
            Number of words in the last user message
        """
        return len(self.user_message().split())
    
    def add_system(self, text: str) -> None:
        """Add text to the system prompt.
        
        Alias for add_system_prompt() with prepend=True.
        
        Args:
            text: Text to add to system prompt
        
        Example::
        
            @app.tag("formal")
            def formal(ctx, tag):
                ctx.add_system("Respond in formal, professional tone.")
        """
        self.add_system_prompt(text, prepend=True)
    
    def set_user_message(self, text: str) -> None:
        """Replace the last user message content.
        
        Args:
            text: New message content
        
        Example::
        
            @app.tag("compress")
            def compress(ctx, tag):
                compressed = " ".join(ctx.user_message().split())
                ctx.set_user_message(compressed)
        """
        for msg in reversed(self.messages):
            if msg.get("role") == "user":
                msg["content"] = text
                return
    
    def add_message(self, message: dict[str, Any]) -> None:
        """Add a message to the conversation history.
        
        Args:
            message: Message dict with 'role' and 'content'
        
        Example::
        
            @app.tag("hint")
            def hint(ctx, tag):
                ctx.add_message({
                    "role": "system",
                    "content": "Provide examples in your response."
                })
        """
        self.messages.append(message)
    
    def cache_key(self) -> str:
        """Generate a cache key for this request.
        
        Returns:
            Hash of the messages (excluding system messages with context)
        
        Example::
        
            @app.tag("cache")
            def check_cache(ctx, tag):
                key = ctx.cache_key()
                if cached := redis.get(key):
                    return ctx.respond(cached)
        """
        import hashlib
        import json
        # Hash messages excluding system (which may have dynamic context)
        msgs = [m for m in self.messages if m.get("role") != "system"]
        content = json.dumps(msgs, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def analyze_complexity(self) -> str:
        """Analyze query complexity using built-in heuristic.
        
        Returns:
            'low', 'medium', or 'high' based on word count
        
        Example::
        
            @app.tag("smart")
            def smart_route(ctx, tag):
                if ctx.analyze_complexity() == "high":
                    return ctx.route_to("margAI/openai/gpt-4o")
                else:
                    return ctx.route_to("margAI/local/llama3.2-1b")
        """
        words = self.user_message_length()
        if words < 20:
            return "low"
        if words < 100:
            return "medium"
        return "high"
    
    # -- Pipeline control ---------------------------------------------------
    
    def route_to(self, model: str) -> RouteOverride:
        """Override routing to a specific model.
        
        Return this from a tag function to force routing to a model.
        
        Args:
            model: A model id in the form ``GET /v1/models`` lists them,
                e.g. ``margAI/local/llama3.2-1b``. An id that does not
                resolve is a 404 naming the ids that do.
        
        Returns:
            RouteOverride object to return from tag function
        
        Example::
        
            @app.tag("fast")
            def fast_route(ctx):
                return ctx.route_to("margAI/local/llama3.2-1b")
        """
        return RouteOverride(model)
    
    def respond(self, payload: dict[str, Any]) -> ImmediateResponse:
        """Skip LLM and return this response immediately.
        
        Return this from a tag function to short-circuit the pipeline.
        
        Args:
            payload: OpenAI-format response dict
        
        Returns:
            ImmediateResponse object to return from tag function
        
        Example::
        
            @app.tag("cache")
            def check_cache(ctx):
                if cached := cache.get(ctx.cache_key()):
                    return ctx.respond(cached)
        """
        return ImmediateResponse(payload)
    
    def error(
        self,
        status: int,
        message: str,
        error_type: str = "invalid_request_error"
    ) -> ErrorResponse:
        """Return an error response immediately.
        
        Return this from a tag function to fail the request.
        
        Args:
            status: HTTP status code
            message: Error message
            error_type: OpenAI error type
        
        Returns:
            ErrorResponse object to return from tag function
        
        Example::
        
            @app.tag("validate")
            def validate(ctx):
                if len(ctx.user_message()) > 10000:
                    return ctx.error(400, "Message too long")
        """
        return ErrorResponse(status, message, error_type)
