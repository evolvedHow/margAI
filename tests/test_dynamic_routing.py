"""Dynamic routing: the selector chain behind the reserved `margAI/dynamic` id."""

from __future__ import annotations

import asyncio

import pytest
from conftest import FakeTransport, UpstreamResponse, chat_payload, make_wrapper, provider_config

from margAI import install_bangtags
from margAI.config import GatewayConfig, TelemetryConfig
from margAI.core.errors import ApiError
from margAI.core.intent import RoutingIntent
from margAI.core.marglets import Marglet
from margAI.core.router import Candidate, Route
from margAI.routing import coerce_route
from margAI.telemetry import Telemetry

MULTI = (
    provider_config(name="openai", models=("gpt-4o",), default_model="gpt-4o"),
    provider_config(
        name="openrouter",
        base_url="http://or.test/v1",
        models=("gpt-4o", "llama3"),
        default_model="llama3",
    ),
    provider_config(name="local", base_url="http://local.test/v1", models=("qwen3",), default_model="qwen3"),
)


def build(*, default_provider=None, telemetry=None, providers=MULTI, responses=None):
    t = FakeTransport(responses=responses or [UpstreamResponse(200, chat_payload("hi"))])
    w = make_wrapper(
        t,
        providers=providers,
        gateway=GatewayConfig(default_provider=default_provider),
        telemetry=telemetry or Telemetry(TelemetryConfig(enabled=False)),
    )
    install_bangtags(w)
    return w, t


def call(w, *, model="margAI/dynamic", tag=None, text="hi", stream=False):
    content = f"{text}  !margAI: {tag}" if tag else text
    return {"model": model, "messages": [{"role": "user", "content": content}], "stream": stream}


def routed_to(t) -> str:
    return f"{t.requested[0].url}"


# -- the chain -------------------------------------------------------------


def test_first_non_none_selector_wins():
    w, t = build()
    w.routing.add_selector(lambda i, c, ctx: "local/qwen3", name="a", order=0)
    w.routing.add_selector(lambda i, c, ctx: "openai/gpt-4o", name="b", order=10)
    r = asyncio.run(w.complete(call(w)))
    assert r.status == 200
    assert "local.test" in routed_to(t)


def test_abstention_falls_through_to_the_next_selector():
    w, t = build()
    w.routing.add_selector(lambda i, c, ctx: None, name="abstains", order=0)
    w.routing.add_selector(lambda i, c, ctx: "openai/gpt-4o", name="claims", order=10)
    r = asyncio.run(w.complete(call(w)))
    assert r.status == 200
    assert "upstream.test" in routed_to(t)


def test_no_selector_falls_back_to_default_provider():
    w, t = build(default_provider="local")
    r = asyncio.run(w.complete(call(w)))
    assert r.status == 200
    assert "local.test" in routed_to(t)


def test_no_selector_and_no_default_is_a_helpful_404():
    w, _ = build()
    with pytest.raises(ApiError) as e:
        asyncio.run(w.routing.select(RoutingIntent()))
    assert e.value.status == 404
    assert "no dynamic selector" in e.value.message
    assert "default_provider" in e.value.message


def test_no_candidates_matching_constraints_is_a_404():
    w, _ = build(default_provider="openai")
    with pytest.raises(ApiError) as e:
        asyncio.run(w.routing.select(RoutingIntent(exclude=frozenset({"openai", "openrouter", "local"}))))
    assert e.value.status == 404
    assert "no configured" in e.value.message


def test_selector_list_is_a_preference_order():
    """A list means "any of these will do": a stale first entry must not
    throw away the rest."""
    w, t = build(default_provider="openai")
    w.routing.add_selector(
        lambda i, c, ctx: ["ghost/phantom", "local/qwen3"], name="prefers", order=0
    )
    r = asyncio.run(w.complete(call(w)))
    assert r.status == 200
    assert "local.test" in routed_to(t)


def test_selector_order_decides_precedence():
    w, t = build()
    w.routing.add_selector(lambda i, c, ctx: "openai/gpt-4o", name="late", order=10)
    w.routing.add_selector(lambda i, c, ctx: "local/qwen3", name="early", order=-10)
    asyncio.run(w.complete(call(w)))
    assert "local.test" in routed_to(t)


def test_registration_order_breaks_order_ties():
    w, t = build()
    w.routing.add_selector(lambda i, c, ctx: "openai/gpt-4o", name="first")
    w.routing.add_selector(lambda i, c, ctx: "local/qwen3", name="second")
    asyncio.run(w.complete(call(w)))
    assert "upstream.test" in routed_to(t)


# -- what selectors may return --------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        Route("local", "qwen3", "explicit"),
        Candidate("local", "qwen3"),
        "local/qwen3",
        {"provider": "local", "model": "qwen3"},
    ],
)
def test_coerce_route_accepts_every_documented_shape(value):
    route = coerce_route(value, selector="s")
    assert route is not None
    assert (route.provider, route.model) == ("local", "qwen3")


def test_coerce_route_abstention():
    assert coerce_route(None) is None
    assert coerce_route([]) is None


@pytest.mark.parametrize("bad", ["no-slash", {"provider": "local"}, 42])
def test_coerce_route_rejects_nonsense_loudly(bad):
    """A silently ignored selector is worse than a loud one."""
    with pytest.raises(ApiError) as e:
        coerce_route(bad, selector="s")
    assert e.value.status == 500
    assert "s" in e.value.message


def test_selector_returning_an_unknown_model_is_rejected_and_chain_continues():
    w, t = build(default_provider="openai")
    w.routing.add_selector(lambda i, c, ctx: "ghost/phantom", name="typo", order=0)
    w.routing.add_selector(lambda i, c, ctx: "local/qwen3", name="good", order=10)
    r = asyncio.run(w.complete(call(w)))
    assert r.status == 200
    assert "local.test" in routed_to(t)


# -- the intent ------------------------------------------------------------


def test_intent_records_tags_without_interpreting_them():
    """The intent is a name-agnostic record. What `route=` *means* is claimed by
    a handler in margAI.builtin_tags, like any other pack's directive -- so
    this layer cannot know a tag vocabulary, and a pack can add to it."""
    intent = RoutingIntent.from_tags(_tags("route=local/qwen3, not=openai, only=llama3, cheap"))
    assert intent.value("route") == "local/qwen3"
    assert intent.value("not") == "openai"
    assert intent.value("only") == "llama3"
    assert intent.values_for("route") == ("local/qwen3",)
    assert intent.has("cheap")
    assert intent.wants("cheap")
    # Nothing is claimed yet: only a handler can fill a typed field.
    assert intent.provider is None
    assert intent.model is None
    assert intent.exclude == frozenset()
    assert intent.require == frozenset()


def test_a_pack_scoped_selector_is_recorded_qualified():
    """`!hr: approve` must stay distinguishable from `!margAI: approve`, or one
    domain's handler would answer another's tag."""
    from margAI.bangtag import Tag

    intent = RoutingIntent.from_tags([Tag("approve", namespace="hr"), Tag("route", value="x")])
    assert "hr:approve" in intent.qualified
    assert "margai:route" in intent.qualified
    assert intent.from_namespace("hr") == ("hr:approve",)
    assert intent.from_namespace("sc") == ()


def test_builtin_pack_fills_the_typed_fields_through_handlers():
    """The same values, reaching the intent the way a real call reaches them:
    parsed, then claimed by the built-in pack's handlers."""
    w, _ = build()
    seen: list[RoutingIntent] = []

    @w.before(order=-40)
    def capture(ctx):
        seen.append(ctx.intent)

    # A concrete model, so the router never consults the constraints: this is
    # about the selectors reaching the intent, not about how they interact.
    # (Combining `route=local/qwen3` with `only=llama3` under `dynamic` is the
    # documented loud 404, covered by its own test below.)
    r = asyncio.run(
        w.complete(call(w, model="margAI/openai/gpt-4o", tag="route=local/qwen3, not=openai, only=llama3, cheap"))
    )
    assert r.status == 200
    intent = seen[-1]
    assert intent.provider == "local"
    assert intent.model == "qwen3"
    assert "openai" in intent.exclude
    assert "llama3" in intent.require
    assert intent.wants("cheap")


def test_constraint_tokens_match_provider_model_or_pair():
    """`not=openai` has to exclude the whole provider, which is what people
    actually write. Matching only the pair or the bare model let it through."""
    provider_only = RoutingIntent(exclude=frozenset({"openai"}))
    assert provider_only.allows("local", "gpt-4o") is True
    assert provider_only.allows("openai", "anything") is False

    pair = RoutingIntent(exclude=frozenset({"openai/gpt-4o"}))
    assert pair.allows("openai", "gpt-4o") is False
    assert pair.allows("openai", "gpt-4o-mini") is True

    model_only = RoutingIntent(require=frozenset({"gpt-4o"}))
    assert model_only.allows("openai", "gpt-4o") is True
    assert model_only.allows("openai", "gpt-4o-mini") is False


def test_intent_is_immutable_and_refinement_returns_a_new_value():
    base = RoutingIntent(tags=("cheap",))
    refined = base.with_(provider="local").note("why", "user asked")
    assert base.provider is None
    assert base.notes == ()
    assert refined.provider == "local"
    assert refined.tags == ("cheap",)
    assert refined.describe()["notes"] == {"why": "user asked"}


def _tags(raw: str):
    from margAI.bangtag import _parse_tags

    return _parse_tags(raw)


def test_bangtag_route_pinning_beats_the_selector_chain():
    """`!margAI: route=local/qwen3` asked for a specific target."""
    w, t = build(default_provider="openai")
    w.routing.add_selector(lambda i, c, ctx: "openrouter/llama3", name="chain", order=-100)
    r = asyncio.run(w.complete(call(w, tag="route=local/qwen3")))
    assert r.status == 200
    assert "local.test" in routed_to(t)


def test_bangtag_pin_is_still_subject_to_the_calls_own_constraints():
    """`route=` states a preference, `not=` states policy. A contradiction
    between them must fail loudly rather than quietly serve the excluded
    target -- the same rule `default_provider` already follows."""
    w, t = build(default_provider="openai")
    w.routing.add_selector(lambda i, c, ctx: "local/qwen3", name="chain", order=-100)
    r = asyncio.run(w.complete(call(w, tag="route=openai/gpt-4o, not=openai")))
    assert r.status == 404
    assert "exclude" in r.body["error"]["message"]
    assert t.requested == []  # nothing went upstream


def test_bangtag_pin_still_works_when_only_constrains_elsewhere():
    """The constraint check must not break the common case: pinning a target
    that the constraints do not mention."""
    w, t = build(default_provider="openai")
    r = asyncio.run(w.complete(call(w, tag="route=local/qwen3, not=openai")))
    assert r.status == 200
    assert "local.test" in routed_to(t)


def test_bangtag_pin_records_its_route_reason():
    """A pinned dynamic call is still a dynamic call, so it must be
    attributable in telemetry like every other one."""
    records: list = []
    w, _ = build(telemetry=Telemetry(TelemetryConfig(enabled=True, emit="callback"), callback=records.append))
    asyncio.run(w.complete(call(w, tag="route=local/qwen3")))
    assert records[0].provider == "local"
    assert records[0].route_reason == "bangtag:route"


def test_bangtag_pin_naming_an_unknown_provider_is_a_loud_404():
    """A typo'd `route=` must not quietly fall through to the chain and serve
    something the caller did not ask for."""
    w, t = build(default_provider="openai")
    w.routing.add_selector(lambda i, c, ctx: "local/qwen3", name="chain", order=0)
    r = asyncio.run(w.complete(call(w, tag="route=nope/nothing")))
    assert r.status == 404
    assert "nope" in r.body["error"]["message"]
    assert t.requested == []


def test_marglet_can_steer_the_intent_before_routing():
    """A before-hook refines the intent, and routing (which runs after every
    before-hook) sees the refinement."""
    w, t = build(default_provider="openai")
    seen: list[dict] = []

    def steer(ctx, tag):
        ctx.steer(provider="local")

    def record_and_first(intent, candidates, ctx):
        seen.append(intent.describe())
        return candidates[0]

    w.routing.add_selector(record_and_first, name="s")
    w.add_marglet(Marglet("local-only", before=steer))
    asyncio.run(w.complete(call(w, tag="local-only")))
    assert seen[0]["provider"] == "local"
    assert "local.test" in routed_to(t)


def test_selector_only_for_scopes_it_to_matching_tags():
    w, _ = build(default_provider="openai")
    called: list[str] = []
    def scoped(intent, candidates, ctx):
        called.append("scoped")
        return "local/qwen3"

    w.routing.add_selector(scoped, name="scoped", only_for=["premium"])
    asyncio.run(w.complete(call(w, tag="premium")))
    assert called == ["scoped"]
    called.clear()
    asyncio.run(w.complete(call(w, tag="basic")))
    assert called == []


def test_marglet_with_select_becomes_a_scoped_selector():
    w, t = build(default_provider="openai")
    seen: list[str] = []

    def pick(intent, cands, ctx):
        seen.append((intent.tags and intent.tags[0]) or "")
        return "local/qwen3"

    w.add_marglet(Marglet("go-local", before=lambda ctx, tag: None, select=pick, order=-5))
    w.routing.add_selector(lambda i, c, ctx: "openrouter/llama3", name="generic", order=0)
    asyncio.run(w.complete(call(w, tag="go-local")))
    assert "local.test" in routed_to(t)
    assert seen == ["go-local"]
    # ...and it must not fire for calls it is not active on.
    asyncio.run(w.complete(call(w, tag="other")))
    assert seen == ["go-local"]


def test_excluded_default_provider_is_an_error_not_a_silent_violation():
    """`not=openai` must not quietly be served by openai."""
    w, t = build(default_provider="openai")
    r = asyncio.run(w.complete(call(w, tag="not=openai")))
    assert r.status == 404
    assert "exclude" in r.body["error"]["message"]
    assert t.requested == []  # nothing went upstream


def test_default_provider_respects_the_constraints_when_allowed():
    w, t = build(default_provider="local")
    asyncio.run(w.complete(call(w, tag="not=openai")))
    assert "local.test" in routed_to(t)


# -- observability ---------------------------------------------------------


def test_telemetry_records_why_the_route_was_chosen():
    records: list = []
    w, _ = build(telemetry=Telemetry(TelemetryConfig(enabled=True, emit="callback"), callback=records.append))
    w.routing.add_selector(lambda i, c, ctx: "local/qwen3", name="prefers-local")
    w.add_marglet(Marglet("terse", before=lambda ctx, tag: None))
    asyncio.run(w.complete(call(w, tag="terse")))
    assert len(records) == 1
    rec = records[0]
    assert rec.provider == "local"
    assert rec.request_model == "margAI/dynamic"
    assert rec.marglets == ("terse",)
    assert "prefers-local" in rec.route_reason


def test_telemetry_records_the_fallback_reason():
    records: list = []
    w, _ = build(
        default_provider="local",
        telemetry=Telemetry(TelemetryConfig(enabled=True, emit="callback"), callback=records.append),
    )
    asyncio.run(w.complete(call(w)))
    assert records[0].route_reason == "default_provider"


def test_dynamic_flag_is_visible_on_the_context():
    w, _ = build(default_provider="local")
    seen: list[bool] = []
    def record_dynamic(intent, candidates, ctx):
        seen.append(ctx.dynamic)
        return "local/qwen3"

    w.routing.add_selector(record_dynamic, name="s")
    asyncio.run(w.complete(call(w)))
    assert seen == [True]


def test_non_dynamic_calls_bypass_the_chain_entirely():
    w, t = build(default_provider="openai")
    called: list[str] = []
    def record_and_route(intent, candidates, ctx):
        called.append("x")
        return "local/qwen3"

    w.routing.add_selector(record_and_route, name="s")
    asyncio.run(w.complete(call(w, model="margAI/openai/gpt-4o")))
    assert called == []
    assert "upstream.test" in routed_to(t)


def test_describe_reports_the_chain():
    w, _ = build(default_provider="local")
    w.routing.add_selector(lambda i, c, ctx: None, name="a", only_for=["x"])
    info = w.routing.describe()
    assert info["dynamic_id"] == "margAI/dynamic"
    assert info["selectors"][0]["name"] == "a"
    assert info["selectors"][0]["only_for"] == ["x"]
    assert "local/qwen3" in info["candidates"]
    assert info["default_provider"] == "local"


def test_remove_selector_returns_the_removed_one():
    w, _ = build(default_provider="local")
    sel = w.routing.add_selector(lambda i, c, ctx: None, name="gone")
    assert len(w.routing) == 1
    assert w.routing.remove_selector("gone") is sel
    assert len(w.routing) == 0
    assert w.routing.remove_selector("gone") is None  # idempotent


# -- streaming -------------------------------------------------------------


def test_stream_preflight_failure_runs_error_hooks_and_records_telemetry():
    """A stream that never opened used to skip both, so a rejected dynamic
    route looked like a call that never happened."""
    records: list = []
    w, _ = build(
        default_provider="openai",
        telemetry=Telemetry(TelemetryConfig(enabled=True, emit="callback"), callback=records.append),
    )
    seen: list[str] = []
    w.add_marglet(Marglet("watch", before=lambda ctx, tag: None, error=lambda exc, ctx, tag: seen.extend(ctx.marglets)))
    with pytest.raises(ApiError):
        asyncio.run(w.open_stream(call(w, tag="watch, not=openai", stream=True)))
    assert seen == ["watch"]
    assert len(records) == 1
    assert records[0].status == 404
    assert records[0].stream is True


def test_dynamic_streaming_call_routes_and_streams():
    w, _ = build(default_provider="local", responses=None)
    import json

    from conftest import FakeStream, chunk_payload

    raw = [f"data: {json.dumps(chunk_payload('x'))}", "data: [DONE]"]
    t = w.transport
    t.streams = [FakeStream(raw)]
    w.routing.add_selector(lambda i, c, ctx: "openai/gpt-4o", name="s")
    handle = asyncio.run(w.open_stream(call(w, stream=True)))
    assert handle is not None
    assert "upstream.test" in t.requested[0].url


def test_dynamic_id_is_advertised_in_the_catalog():
    """A client picking from a model dropdown can only use the dynamic id if
    it can see it."""
    w, _ = build(default_provider="local")
    ids = [m["id"] for m in asyncio.run(w.models())]
    assert "margAI/dynamic" in ids
    assert "margAI/local/qwen3" in ids  # namespaced, per `expose`


def test_fastapi_surface_serves_a_dynamic_call():
    from starlette.testclient import TestClient

    from margAI.transport.fastapi import build_app

    records: list = []
    t = FakeTransport(responses=[UpstreamResponse(200, chat_payload("routed"))])
    w = make_wrapper(
        t,
        providers=MULTI,
        gateway=GatewayConfig(default_provider="openai"),
        telemetry=Telemetry(TelemetryConfig(enabled=True, emit="callback"), callback=records.append),
    )
    install_bangtags(w)
    w.routing.add_selector(lambda i, c, ctx: "local/qwen3", name="prefer-local")
    w.add_marglet(Marglet("terse", before=lambda ctx, tag: ctx.add_system_prompt("Be brief.")))

    with TestClient(build_app(w)) as client:
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "margAI/dynamic", "messages": [{"role": "user", "content": "hi  !margAI: terse"}]},
        )
    assert resp.status_code == 200
    assert "local.test" in t.requested[0].url
    assert t.requested[0].json["messages"][0]["role"] == "system"
    assert records[0].marglets == ("terse",)
    assert "prefer-local" in records[0].route_reason
