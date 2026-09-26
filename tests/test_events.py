"""EventRegistry dispatch and the Wrapper's FastAPI-style event decorators."""

from __future__ import annotations

import asyncio
import json

from conftest import FakeStream, FakeTransport, chat_payload, chunk_payload, make_wrapper

from margAI import EventHandler, EventRegistry, Tag, install_bangtags
from margAI.core.protocol import UpstreamResponse


class _Ctx:
    def __init__(self) -> None:
        self.state: dict = {}


def test_registry_dispatch_before_after():
    registry = EventRegistry()
    calls: list[str] = []

    @registry.before("demo")
    def demo_before(ctx, tag):
        calls.append(f"before:{tag.name}:{tag.value}")

    @registry.after("demo")
    def demo_after(payload, ctx, tag):
        calls.append("after")
        return payload

    async def go():
        await registry.run_before([Tag("demo", "x")], _Ctx())
        return await registry.run_after([Tag("demo", "x")], {"a": 1}, _Ctx())

    assert asyncio.run(go()) == {"a": 1}
    assert calls == ["before:demo:x", "after"]


def test_handlers_may_be_async():
    registry = EventRegistry()

    @registry.before("slow")
    async def slow_before(ctx, tag):
        ctx.state["saw"] = tag.name

    async def go():
        ctx = _Ctx()
        await registry.run_before([Tag("slow")], ctx)
        return ctx.state["saw"]

    assert asyncio.run(go())


def test_registry_dispatch_unknown_is_noop():
    registry = EventRegistry()
    assert registry.all() == set()
    assert registry.describe() == {}

    async def go():
        await registry.run_before([Tag("missing")], _Ctx())
        return await registry.run_after([Tag("missing")], {"a": 1}, _Ctx())

    assert asyncio.run(go()) == {"a": 1}


def test_stream_dispatch_and_drop_chunk():
    registry = EventRegistry()

    @registry.stream("boom")
    def boom(chunk, ctx, tag, scratch):
        if chunk.get("choices", [{}])[0].get("delta", {}).get("content") == "b":
            return None
        return chunk

    async def go():
        kept = await registry.run_stream([Tag("boom")], {"choices": [{"delta": {"content": "a"}}]}, _Ctx(), {})
        dropped = await registry.run_stream([Tag("boom")], {"choices": [{"delta": {"content": "b"}}]}, _Ctx(), {})
        return kept, dropped

    kept, dropped = asyncio.run(go())
    assert kept == {"choices": [{"delta": {"content": "a"}}]}
    assert dropped is None


def test_registry_describe_docs():
    registry = EventRegistry()

    @registry.before("doced")
    def fn(ctx, tag):
        """First line of docs."""

    assert registry.describe()["doced"] == "First line of docs."


def test_describe_prefers_first_registered_event():
    registry = EventRegistry()

    @registry.before("both")
    def before_both(ctx, tag):
        """before doc"""

    @registry.after("both")
    def after_both(payload, ctx, tag):
        """after doc"""

    assert registry.describe()["both"] == "before doc"
    assert registry.all() == {"both"}


def test_error_dispatch_returns_first_handler_dict():
    registry = EventRegistry()

    @registry.error("boom")
    def boom(exc, ctx, tag):
        return {"error": {"message": str(exc), "type": "tagged"}}

    async def go():
        return await registry.run_error([Tag("boom")], RuntimeError("nope"), _Ctx())

    assert asyncio.run(go()) == {"error": {"message": "nope", "type": "tagged"}}
    assert registry.all() == {"boom"}


def test_event_handler_base_class_method_names():
    registry = EventRegistry()

    class Handlers(EventHandler):
        def before_demo(self, ctx, tag):
            ctx.state["handled"] = tag.name

        def after_demo(self, payload, ctx, tag):
            payload["ok"] = True
            return payload

    registry.add(Handlers())

    assert registry.all() == {"demo"}

    ctx = _Ctx()

    async def go():
        await registry.run_before([Tag("demo")], ctx)
        return await registry.run_after([Tag("demo")], {"a": 1}, ctx)

    assert asyncio.run(go()) == {"a": 1, "ok": True}
    assert ctx.state == {"handled": "demo"}


def _contents(handle_lines):
    out: list[str] = []
    for line in handle_lines:
        if line.strip() == "data: [DONE]" or not line.startswith("data: "):
            continue
        payload = json.loads(line[len("data: ") :])
        choices = payload.get("choices") or []
        content = choices[0].get("delta", {}).get("content")
        if isinstance(content, str):
            out.append(content)
    return out


def _boom(req):
    raise RuntimeError("kaboom")


# ---------------------------------------------------------------------------
# FastAPI-style surface on the Wrapper itself
# ---------------------------------------------------------------------------


def test_wrapper_event_decorators_fire_automatically():
    wrapper = make_wrapper(FakeTransport(responses=[UpstreamResponse(200, chat_payload("hi"))]))

    @wrapper.before("shout")
    def shout_before(ctx, tag):
        ctx.state["saw"] = tag.name

    @wrapper.after("shout")
    def shout_after(payload, ctx, tag):
        payload["choices"][0]["message"]["content"] += "!"
        return payload

    @wrapper.before(order=-100)
    def tag_the_call(ctx):
        ctx.state["bangtags"] = [Tag("shout")]

    async def go():
        return await wrapper.complete({"model": "margAI/openai/gpt-4o", "messages": []})

    res = asyncio.run(go())
    assert res.status == 200
    assert res.body["choices"][0]["message"]["content"] == "hi!"
    assert wrapper.events.all() == {"shout"}


def test_wrapper_events_describe_docs():
    wrapper = make_wrapper(FakeTransport())

    @wrapper.before("doced")
    def fn(ctx, tag):
        """Custom docs line."""

    assert wrapper.events.describe()["doced"] == "Custom docs line."


def test_install_bangtags_end_to_end():
    transport = FakeTransport(responses=[UpstreamResponse(200, chat_payload("hi"))])
    wrapper = make_wrapper(transport)
    install_bangtags(wrapper)

    @wrapper.after("shout")
    def shout(payload, ctx, tag):
        payload["choices"][0]["message"]["content"] += "!"
        return payload

    async def go():
        res = await wrapper.complete(
            {"model": "margAI/openai/gpt-4o", "messages": [{"role": "user", "content": "x !margAI: shout"}]}
        )
        return res, transport.requested[-1].json

    res, upstream = asyncio.run(go())
    assert res.body["choices"][0]["message"]["content"] == "hi!"
    assert "!margAI" not in upstream["messages"][-1]["content"]


def test_wrapper_add_handler_uses_event_handler_group():
    wrapper = make_wrapper(FakeTransport(responses=[UpstreamResponse(200, chat_payload("hi"))]))

    class Group(EventHandler):
        def before_tick(self, ctx, tag):
            ctx.state["before"] = tag.name

        def after_tock(self, payload, ctx, tag):
            payload["choices"][0]["message"]["content"] += "?"
            return payload

    wrapper.add_handler(Group())

    @wrapper.before(order=-100)
    def tag_the_call(ctx):
        ctx.state["bangtags"] = [Tag("tick"), Tag("tock")]

    async def go():
        return await wrapper.complete({"model": "margAI/openai/gpt-4o", "messages": []})

    res = asyncio.run(go())
    assert res.body["choices"][0]["message"]["content"] == "hi?"


def test_wrapper_stream_and_error_events():
    wrapper = make_wrapper(
        FakeTransport(
            responses=[_boom],
            streams=[FakeStream([f"data: {json.dumps(chunk_payload('ab'))}", "data: [DONE]"])],
        )
    )

    @wrapper.stream("mark")
    def stream_mark(chunk, ctx, tag, scratch):
        if not scratch.get("marked") and chunk["choices"][0]["delta"].get("content"):
            scratch["marked"] = True
            chunk["choices"][0]["delta"]["content"] = "*" + chunk["choices"][0]["delta"]["content"]
        return chunk

    @wrapper.error("mark")
    def err(exc, ctx, tag):
        return {"error": {"message": str(exc), "type": "tagged"}}

    @wrapper.before(order=-100)
    def tag_the_call(ctx):
        ctx.state["bangtags"] = [Tag("mark")]

    async def go():
        handle = await wrapper.open_stream({"model": "margAI/openai/gpt-4o", "messages": []})
        deltas = _contents([line async for line in handle.lines()])
        err_res = await wrapper.complete({"model": "margAI/openai/gpt-4o", "messages": []})
        return deltas, err_res

    deltas, err_res = asyncio.run(go())
    assert deltas == ["*ab"]
    assert err_res.body == {"error": {"message": "kaboom", "type": "tagged"}}