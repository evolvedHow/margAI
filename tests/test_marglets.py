"""Marglets: named, bangtag-activated enhancements across all four phases."""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import (
    FakeStream,
    FakeTransport,
    UpstreamResponse,
    chat_payload,
    chunk_payload,
    make_wrapper,
)

from margAI import install_bangtags
from margAI.core.marglets import Marglet, MargletRegistry


def sse_lines(*texts: str) -> list[str]:
    return [f"data: {json.dumps(chunk_payload(t))}" for t in texts] + ["data: [DONE]"]


async def collect(handle) -> list[str]:
    return [line async for line in handle.lines()]


def chat(content: str = "hello", *, model: str = "margAI/openai/gpt-4o", tag: str | None = None):
    text = f"{content}  !margAI: {tag}" if tag else content
    return {"model": model, "messages": [{"role": "user", "content": text}]}


def ok(chat_body=None, status=200):
    return UpstreamResponse(status, chat_payload("hi") or {"error": {"message": "boom"}})


def ok_wrapper(responses=None, streams=None, **kw):
    t = FakeTransport(responses=responses or [UpstreamResponse(200, chat_payload("hi"))], streams=streams)
    w = make_wrapper(t, **kw)
    install_bangtags(w)
    return w, t


# -- registration ----------------------------------------------------------


def test_decorator_form_registers_and_fires():
    w, t = ok_wrapper()

    @w.marglet("terse", summary="Be brief.")
    def terse_before(ctx, tag):
        ctx.add_system_prompt("Be brief.")

    r = asyncio.run(w.complete(chat(tag="terse")))
    assert r.status == 200
    assert t.requested[0].json["messages"][0] == {"role": "system", "content": "Be brief."}
    assert "terse" in w.marglets


def test_inactive_marglet_does_not_fire():
    w, t = ok_wrapper()

    @w.marglet("terse")
    def terse_before(ctx, tag):
        ctx.add_system_prompt("Be brief.")

    asyncio.run(w.complete(chat(tag="something-else")))
    assert all(m["role"] != "system" for m in t.requested[0].json["messages"])


def test_all_four_phases_fire():
    w, _ = ok_wrapper()
    seen: list[str] = []

    def mark(phase):
        """A `before` handler: record, change nothing."""

        def handler(ctx, tag):
            seen.append(phase)

        return handler

    def pass_through(phase):
        """An `after`/`stream` handler: record, return the payload unchanged."""

        def handler(payload, ctx, *rest):
            seen.append(phase)
            return payload

        return handler

    w.add_marglet(
        Marglet(
            "full",
            before=mark("before"),
            after=pass_through("after"),
            stream=pass_through("stream"),
            error=lambda exc, ctx, tag: mark("error")(ctx, tag),
        )
    )

    asyncio.run(w.complete(chat(tag="full")))
    assert seen == ["before", "after"]


def test_duplicate_name_raises():
    w, _ = ok_wrapper()
    w.add_marglet(Marglet("dup", before=lambda ctx, tag: None))
    with pytest.raises(ValueError, match="already registered"):
        w.add_marglet(Marglet("dup", before=lambda ctx, tag: None))
    # ...unless the caller means it.
    w.add_marglet(Marglet("dup", before=lambda ctx, tag: None), replace=True)


def test_add_marglet_group():
    class Bullets:
        def terse_before(self, ctx, tag):
            ctx.add_system_prompt("Be brief.")

        def terse_after(self, payload, ctx, tag):
            payload["choices"][0]["message"]["content"] += " (terse)"
            return payload

        def shout_after(self, payload, ctx, tag):
            payload["choices"][0]["message"]["content"] += "!"
            return payload

    w, _ = ok_wrapper()
    w.add_marglet_group(Bullets())
    assert w.marglets.names() == ["shout", "terse"]
    r = asyncio.run(w.complete(chat(tag="terse")))
    assert r.body["choices"][0]["message"]["content"] == "hi (terse)"


def test_stream_phase_transforms_chunks():
    w, _ = ok_wrapper(streams=[FakeStream(sse_lines("a", "b"))])
    w.add_marglet(
        Marglet("caps", stream=lambda chunk, ctx, tag, scratch: {
            **chunk,
            "choices": [
                {**c, "delta": {**c["delta"], "content": c["delta"].get("content", "").upper()}}
                for c in chunk["choices"]
            ],
        })
    )
    handle = asyncio.run(w.open_stream({**chat(tag="caps"), "stream": True}))
    lines = asyncio.run(collect(handle))
    contents = [
        json.loads(ln[6:])["choices"][0]["delta"]["content"]
        for ln in lines
        if ln.startswith("data: ") and "[DONE]" not in ln
    ]
    assert contents == ["A", "B"]


def test_error_phase_takes_over_body():
    w, _ = ok_wrapper(responses=[UpstreamResponse(500, {"error": {"message": "boom"}})])
    w.add_marglet(
        Marglet(
            "friendly",
            error=lambda exc, ctx, tag: {
                "error": {"message": "try again later", "type": "server_error"}
            },
        )
    )
    r = asyncio.run(w.complete(chat(tag="friendly")))
    assert r.status == 500
    assert r.body["error"]["message"] == "try again later"


def test_marglets_visible_in_success_phase():
    """`ctx.marglets` has to be populated before the handlers run, in every
    phase it can appear in."""
    w, _ = ok_wrapper()
    seen: dict[str, object] = {}

    def snap(key):
        def hook(*a):
            ctx = next(x for x in a if hasattr(x, "marglets"))
            seen[key] = ctx.marglets
            return

        return hook

    w.add_marglet(Marglet("watch", before=snap("before"), after=snap("after")))
    asyncio.run(w.complete(chat(tag="watch")))
    assert seen == {"before": ["watch"], "after": ["watch"]}


def test_marglets_visible_in_error_phase():
    """On a failure the error phase is the only record of what was asked
    for, so `ctx.marglets` has to be populated there too."""
    w, _ = ok_wrapper(responses=[UpstreamResponse(500, {"error": {"message": "boom"}})])
    seen: dict[str, object] = {}

    def snap(*a):
        ctx = next(x for x in a if hasattr(x, "marglets"))
        seen["error"] = list(ctx.marglets)

    w.add_marglet(Marglet("watch", before=lambda *a: seen.__setitem__("before", "fired"), error=snap))
    asyncio.run(w.complete(chat(tag="watch")))
    assert seen == {"before": "fired", "error": ["watch"]}


def test_marglets_recorded_even_when_a_before_hook_raises():
    """A failing marglet must not make the call unattributable."""
    w, _ = ok_wrapper(responses=[UpstreamResponse(200, chat_payload("hi"))])
    seen: list[str] = []

    def boom(ctx, tag):
        raise RuntimeError("nope")

    w.add_marglet(Marglet("watch", before=boom, error=lambda exc, ctx, tag: seen.extend(ctx.marglets)))
    r = asyncio.run(w.complete(chat(tag="watch")))
    assert r.status == 500
    assert seen == ["watch"]


def test_order_controls_dispatch_sequence():
    w, _ = ok_wrapper()
    calls: list[str] = []
    w.add_marglet(Marglet("late", order=10, before=lambda ctx, tag: calls.append("late")))
    w.add_marglet(Marglet("early", order=-10, before=lambda ctx, tag: calls.append("early")))
    w.add_marglet(Marglet("mid", order=0, before=lambda ctx, tag: calls.append("mid")))
    asyncio.run(w.complete(chat(tag="late, early, mid")))
    assert calls == ["early", "mid", "late"]


def test_bangtag_directive_is_stripped_before_upstream():
    w, t = ok_wrapper()
    w.add_marglet(Marglet("terse", before=lambda ctx, tag: None))
    asyncio.run(w.complete(chat("explain", tag="terse")))
    assert t.requested[0].json["messages"][0]["content"] == "explain"


# -- registry (pure, no wrapper) -------------------------------------------


def test_registry_lookup_rules():
    reg = MargletRegistry()
    reg.add(Marglet("a"))
    found = reg.get("a")
    assert "a" in reg and found is not None and found.name == "a"
    assert reg.get("missing") is None
    assert len(reg) == 1
    assert [m.name for m in reg] == ["a"]
    assert reg.remove("a") is not None
    assert reg.remove("a") is None


def test_registry_active_filters_and_dedupes():
    reg = MargletRegistry()
    reg.add(Marglet("a", order=0))
    reg.add(Marglet("b", order=-1))
    reg.add(Marglet("c", order=5))
    # "a" twice, "b" once, and an unregistered tag -> only registered ones
    from margAI import Tag

    active = reg.active([Tag("a"), Tag("b"), Tag("a"), Tag("unknown")])
    assert [m.name for m in active] == ["b", "a"]


def test_registry_specs_and_describe():
    reg = MargletRegistry()
    reg.add(Marglet("a", summary="does a", before=lambda ctx, tag: None))
    assert reg.describe() == {"a": "does a"}
    assert reg.specs()[0]["name"] == "a"
    assert reg.specs()[0]["summary"] == "does a"


def test_marglet_needs_a_name():
    with pytest.raises(ValueError, match="needs a name"):
        Marglet("")


def test_marglet_spec_reports_routing_affinity():
    plain = Marglet("plain", before=lambda ctx, tag: None)
    router = Marglet("router", select=lambda intent, cands, ctx: None)
    assert plain.affects_routing is False
    assert router.affects_routing is True
    assert router.spec().affects_routing is True
    assert "select" in repr(router)


def test_bangtag_install_is_idempotent_per_namespace():
    w, _ = ok_wrapper()

    install_bangtags(w)
    install_bangtags(w)  # same namespace again: no-op
    install_bangtags(w, namespace="acme")  # a different one must still register

    # install_bangtags registers a per-call request hook, so look there.
    names = [h.name for h in w._hooks.requests()]
    assert names.count("bangtags:margai") == 1
    assert names.count("bangtags:acme") == 1
    # The guard must not leave bookkeeping on the object it was handed.
    assert not hasattr(w, "_bangtags_installed")
