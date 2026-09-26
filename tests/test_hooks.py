"""Hook registry, ordering, and decorator semantics."""

from __future__ import annotations

from margAI.core import HookKind, HookRegistry, RequestContext, maybe_await
from margAI.core.hooks import Hook


def ctx(body=None):
    return RequestContext(body=body or {"messages": []})


def test_request_hooks_run_forward():
    reg = HookRegistry()
    order = []

    async def go():
        for h in reg.requests():
            await maybe_await(h.fn(ctx()))
            order.append(h.name)

    reg.register(HookKind.REQUEST, lambda c: None, "r1", 0)
    reg.register(HookKind.REQUEST, lambda c: None, "r2", 0)

    import asyncio

    asyncio.run(go())
    assert order == ["r1", "r2"]


def test_response_hooks_run_in_reverse():
    reg = HookRegistry()
    order: list[str] = []

    def first(payload, ctx):
        order.append("s1")
        return payload

    def second(payload, ctx):
        order.append("s2")
        return payload

    reg.register(HookKind.RESPONSE, first, "s1", 0)
    reg.register(HookKind.RESPONSE, second, "s2", 0)

    async def go():
        return await reg.apply_response({"a": 1}, ctx())

    import asyncio

    asyncio.run(go())
    assert order == ["s2", "s1"]


def test_stream_hook_can_drop_a_chunk():
    reg = HookRegistry()
    reg.register(HookKind.STREAM, lambda chunk, c: None if chunk["n"] == 2 else chunk, "drop2", 0)

    async def go():
        return await reg.apply_stream({"n": 2, "choices": []}, ctx())

    import asyncio

    assert asyncio.run(go()) is None


def test_explicit_order_shifts_relative_position():
    reg = HookRegistry()
    order = []
    reg.register(HookKind.REQUEST, lambda c: order.append("default"), "default", 0)
    reg.register(HookKind.REQUEST, lambda c: order.append("early"), "early", -1)

    async def go():
        for h in reg.requests():
            await maybe_await(h.fn(ctx()))

    import asyncio

    asyncio.run(go())
    assert order == ["early", "default"]


def test_stream_hooks_run_in_reverse():
    reg = HookRegistry()
    order: list[str] = []

    def first(chunk, ctx):
        order.append("a")
        return chunk

    def second(chunk, ctx):
        order.append("b")
        return chunk

    reg.register(HookKind.STREAM, first, "a", 0)
    reg.register(HookKind.STREAM, second, "b", 0)

    async def go():
        await reg.apply_stream({"choices": []}, ctx())

    import asyncio

    asyncio.run(go())
    assert order == ["b", "a"]


def test_maybe_await_handles_sync_and_async():
    async def go():
        assert await maybe_await(1) == 1
        assert await maybe_await(_async()) == 5

    async def _async():
        return 5

    import asyncio

    asyncio.run(go())


def test_registry_counts():
    reg = HookRegistry()
    reg.register(HookKind.REQUEST, lambda c: None, "a")
    reg.register(HookKind.STREAM, lambda c, x: None, "b")
    assert reg.count(HookKind.REQUEST) == 1
    assert reg.count() == 2


def test_hook_dataclass_fields():
    reg = HookRegistry()
    hook = reg.register(HookKind.REQUEST, lambda c: None, "named", order=3)
    assert hook.name == "named"
    assert hook.order == 3
    assert isinstance(hook, Hook)