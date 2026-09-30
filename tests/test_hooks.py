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

# -- cached ordering ------------------------------------------------------
#
# The per-kind orderings are computed at registration and reused, because
# `apply_stream` asks for them once per SSE chunk. These pin the behaviour
# that caching must not have changed.


def test_cached_order_matches_a_fresh_sort_of_every_hook():
    """The cache is only safe if it agrees with sorting the whole list. Walk
    an adversarial mix of orders and kinds, and check both directions."""

    def fresh(reg, kind, reverse):
        hooks = [h for h in reg._hooks if h.kind is kind]
        hooks.sort(key=lambda h: (h.order, h.seq), reverse=reverse)
        return [h.name for h in hooks]

    orders = [-100, -50, -5, 0, 0, 0, 7]
    reg = HookRegistry()
    for i in range(len(orders)):
        for kind in HookKind:
            reg.register(kind, lambda c: None, f"{kind.value}{i}", orders[i])

    reversed_accessors = {
        HookKind.RESPONSE: reg.responses,
        HookKind.STREAM: reg.streams,
        HookKind.ERROR: reg.errors,
    }
    for kind in HookKind:
        assert [h.name for h in reg._ordered[kind]] == fresh(reg, kind, False)
        accessor = reversed_accessors.get(kind)
        if accessor is not None:
            assert [h.name for h in accessor()] == fresh(reg, kind, True)


def test_a_later_hook_can_still_sort_before_an_earlier_one():
    """Registration appends in O(1) when the new hook sorts last. This is the
    case that must still trigger a reorder of the bucket."""
    reg = HookRegistry()
    reg.register(HookKind.REQUEST, lambda c: None, "tail", 0)
    reg.register(HookKind.REQUEST, lambda c: None, "head", -100)
    reg.register(HookKind.REQUEST, lambda c: None, "middle", -50)
    assert [h.name for h in reg.requests()] == ["head", "middle", "tail"]


def test_hooks_of_different_kinds_do_not_interleave():
    reg = HookRegistry()
    reg.register(HookKind.STREAM, lambda c: None, "s1")
    reg.register(HookKind.REQUEST, lambda c: None, "r1")
    reg.register(HookKind.STREAM, lambda c: None, "s2", -10)
    assert [h.name for h in reg.requests()] == ["r1"]
    # Streams come back reversed (middleware unwind): s1 was registered first
    # but s2 was registered at a lower order, so s1 still unwinds first.
    assert [h.name for h in reg.streams()] == ["s1", "s2"]
    assert [h.name for h in reg._ordered[HookKind.STREAM]] == ["s2", "s1"]


def test_the_accessors_do_not_hand_out_internal_state():
    """`responses()` and friends reverse the cached list, so a caller that
    mutates the result must not corrupt the registry."""
    reg = HookRegistry()
    for name in ("a", "b", "c"):
        reg.register(HookKind.STREAM, lambda c: None, name)
    reg.streams().clear()
    reg.streams().append(Hook(HookKind.STREAM, lambda c: None, "injected"))
    assert [h.name for h in reg.streams()] == ["c", "b", "a"]
