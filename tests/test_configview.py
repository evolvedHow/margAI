"""Per-call configuration, and the line it refuses to cross.

Most of this file is about what a per-call config may *not* say. A per-call
config is remote input, and the tempting way to make it powerful -- let it set
any config key -- would let "send a request" become "run a module on the
gateway". The whitelist is the security property; the merge semantics are
convenience.
"""

from __future__ import annotations

import asyncio

import pytest
from conftest import FakeTransport, make_wrapper, provider_config

from margAI.config import Config, GatewayConfig, ModelPolicy
from margAI.core.configview import (
    EMPTY_CONFIG,
    ConfigView,
    ConfigViolation,
    deep_merge,
    safe_overlay,
)

# -- the whitelist ----------------------------------------------------------


def test_a_pack_setting_is_allowed():
    view = safe_overlay("[packs.audit]\nthreshold = 0.5\n")
    assert view.pack("audit", "threshold") == 0.5


def test_a_virtual_model_override_is_allowed():
    view = safe_overlay('[models.fast]\nmax_cost_per_1m = 1.0\n')
    assert view.model_policy("fast")["max_cost_per_1m"] == 1.0


@pytest.mark.parametrize(
    "toml_text",
    [
        '[hooks]\nload = ["os.system"]',
        '[hooks]\nload = "evil"',
        '[providers.evil]\nbase_url = "http://attacker.test"',
        '[telemetry]\ncallback = "attacker:exfil"',
        '[gateway]\nprefix = "other"',
    ],
)
def test_a_config_that_names_code_or_endpoints_is_refused(toml_text):
    """Each of these turns a request into a capability: an import, an
    outbound URL, a callback the gateway will invoke on the caller's behalf."""
    with pytest.raises(ConfigViolation) as exc:
        safe_overlay(toml_text)
    assert "not settable per call" in str(exc.value)


@pytest.mark.parametrize("key", ["hooks", "load", "import", "module", "url", "endpoint", "callback"])
def test_pack_keys_that_name_code_are_refused(key):
    with pytest.raises(ConfigViolation) as exc:
        safe_overlay(f"[packs.audit]\n{key} = 1\n")
    assert "not settable" in str(exc.value)


def test_a_refusal_names_the_offending_key():
    with pytest.raises(ConfigViolation, match=r"\[packs\.audit\.hooks\]"):
        safe_overlay("[packs.audit]\nhooks = 1\n")


def test_the_refusal_lists_what_is_settable():
    """A caller debugging their own request needs the shape of the hole, not
    just the fact of it."""
    with pytest.raises(ConfigViolation) as exc:
        safe_overlay("[gateway]\nport = 1\n")
    assert "packs" in str(exc.value)
    assert "models" in str(exc.value)


def test_a_packs_entry_must_be_a_table():
    with pytest.raises(ConfigViolation, match=r"\[packs\.audit\] must be a table"):
        safe_overlay("[packs]\naudit = 3\n")


def test_invalid_toml_is_rejected_rather_than_ignored():
    with pytest.raises(ConfigViolation, match="not valid TOML"):
        safe_overlay("this is not toml = = =")


# -- merge semantics --------------------------------------------------------


def test_tables_merge_deeply():
    merged = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3, "z": 4}})
    assert merged == {"a": {"x": 1, "y": 3, "z": 4}}


def test_lists_replace_rather_than_merge():
    """Merging lists would produce a list where the pack expects a string --
    a type error surfacing as a 500 on a request the caller configured
    correctly."""
    merged = deep_merge({"options": ["a", "b"]}, {"options": ["c"]})
    assert merged == {"options": ["c"]}


def test_deep_merge_guards_against_runaway_nesting():
    """Counted on the result, not on the merge's own recursion: an already-deep
    value assigned wholesale triggers no recursive call, so a check that only
    counted recursion would check nothing."""
    deep: dict = {"a": 1}
    for _ in range(20):
        deep = {"a": deep}
    with pytest.raises(ConfigViolation, match="too deeply"):
        deep_merge({}, deep)


def test_deep_merge_merges_shallow_tables_fine():
    assert deep_merge({"a": {"b": 1}}, {"a": {"c": 2}}) == {"a": {"b": 1, "c": 2}}


# -- lookup -----------------------------------------------------------------


def test_a_missing_pack_key_returns_the_default():
    view = safe_overlay("[packs.audit]\nthreshold = 0.5\n")
    assert view.pack("audit", "missing", "fallback") == "fallback"
    assert view.pack("nosuchpack", "threshold") is None


def test_enabled_defaults_to_on():
    """A pack is installed because the operator wanted it, so the caller's
    silence is not a request to disable it."""
    assert ConfigView().enabled("audit") is True
    assert safe_overlay("[packs.audit]\nenabled = false\n").enabled("audit") is False


def test_dotted_lookup():
    view = safe_overlay("[packs.audit]\nthreshold = 0.5\n")
    assert view.get("packs.audit.threshold") == 0.5
    assert view.get("packs.audit.nope", "d") == "d"
    assert view.get("nope.nope.nope", "d") == "d"


def test_a_view_is_a_copy_not_a_live_reference():
    """Handing a pack the caller's nested dict would let it mutate a structure
    another pack is reading mid-request."""
    original = {"packs": {"audit": {"prompt": {"text": "hi"}}}, "models": {}}
    view = safe_overlay(original)
    original["packs"]["audit"]["prompt"]["text"] = "tampered"
    assert view.pack("audit", "prompt") == {"text": "hi"}


def test_as_dict_is_round_trippable():
    view = safe_overlay("[packs.audit]\nthreshold = 0.5\n")
    as_dict = view.as_dict()
    assert as_dict["packs"]["audit"]["threshold"] == 0.5
    assert as_dict["source"] == "call"


def test_the_empty_view_is_shared_and_safe_to_hand_out():
    assert EMPTY_CONFIG.packs == {}
    assert EMPTY_CONFIG.source == "default"


# -- through the gateway ----------------------------------------------------


def test_a_handler_reads_its_own_per_call_config():
    app = make_wrapper(FakeTransport())
    seen: list[object] = []

    @app.before
    def read_config(ctx):
        seen.append(ctx.config.pack("audit", "threshold"))

    asyncio.run(
        app.complete(
            {"model": "margAI/openai/gpt-4o", "messages": []},
            config="[packs.audit]\nthreshold = 0.5\n",
        )
    )
    assert seen == [0.5]


def test_deployment_config_is_the_default_and_a_call_may_override_it():
    app = make_wrapper(FakeTransport())
    seen: list[object] = []

    @app.before
    def read_config(ctx):
        seen.append(ctx.config.pack("audit", "threshold"))

    body = {"model": "margAI/openai/gpt-4o", "messages": []}
    asyncio.run(app.complete(body))
    asyncio.run(app.complete(body, config="[packs.audit]\nthreshold = 0.5\n"))
    assert seen == [None, 0.5]


def test_a_refused_per_call_config_is_a_400_and_never_reaches_a_handler():
    app = make_wrapper(FakeTransport())
    ran: list[int] = []

    @app.before
    def note(ctx):
        ran.append(1)

    result = asyncio.run(
        app.complete(
            {"model": "margAI/openai/gpt-4o", "messages": []},
            config='[hooks]\nload = ["os.system"]',
        )
    )
    assert result.status == 400
    assert "not settable per call" in result.body["error"]["message"]
    assert ran == [], "a rejected config must not have run any handler"


def test_a_streaming_call_sees_the_same_config():
    """A `before` pack and a `stream` pack must read one view, not two."""
    from conftest import FakeStream, sse_line

    app = make_wrapper(FakeTransport(streams=[FakeStream([sse_line({"id": "1"}), sse_line({"id": "2"})])]))
    seen: list[object] = []

    @app.before
    def read_config(ctx):
        seen.append(ctx.config.pack("audit", "threshold"))

    @app.stream
    def read_config_again(chunk, ctx, scratch):
        seen.append(ctx.config.pack("audit", "threshold"))

    body = {"model": "margAI/openai/gpt-4o", "messages": [], "stream": True}

    async def drain() -> None:
        handle = await app.open_stream(body, config="[packs.audit]\nthreshold = 0.9\n")
        async for _line in handle.lines():
            pass

    asyncio.run(drain())
    assert seen, "the stream never ran"
    assert set(seen) == {0.9}


def test_a_per_call_model_override_actually_changes_routing():
    """The regression this guards: the whitelist accepted `[models.*]`, the
    overlay reached `ctx.config`, and the router ignored it -- a config that
    validated and did nothing."""
    transport = FakeTransport()
    app = make_wrapper(
        transport,
        providers=(
            provider_config("cheap", models=("tiny",)),
            provider_config("dear", models=("huge",)),
        ),
        # `first` with an explicit allow-list, so the baseline does not depend
        # on candidate ordering -- the test is about the override, not about
        # which of two models sorts first.
        models={"pick": ModelPolicy(name="pick", strategy="first", providers=("cheap",))},
    )

    # Read the model that reached the wire, not the one the wrapper reports:
    # "what the policy decided" and "what the provider was sent" are exactly
    # the two things a bug would let drift apart.
    def wire_model(config=None):
        asyncio.run(
            app.complete(
                {"model": "margAI/pick", "messages": [{"role": "user", "content": "hi"}]},
                config=config,
            )
        )
        return transport.requested[-1].json["model"]

    assert wire_model() == "tiny"
    assert wire_model({"models": {"pick": {"providers": ["dear"]}}}) == "huge"


def test_a_per_call_override_that_restates_the_deployment_keeps_the_round_robin_cursor():
    """Rebuilding the resolver per call resets the cursor, so every call gets
    the same target -- a policy that looks like it is alternating and is not."""
    transport = FakeTransport()
    app = make_wrapper(
        transport,
        providers=(provider_config("p", models=("a", "b")),),
        models={"rr": ModelPolicy(name="rr", strategy="round_robin")},
    )
    same = {"models": {"rr": {"strategy": "round_robin"}}}

    def pick():
        asyncio.run(
            app.complete(
                {"model": "margAI/rr", "messages": [{"role": "user", "content": "hi"}]},
                config=same,
            )
        )
        return transport.requested[-1].json["model"]

    assert [pick() for _ in range(4)] == ["a", "b", "a", "b"]


def test_a_per_call_pack_override_deep_merges_rather_than_replacing():
    """A top-level merge would drop the operator's other keys, leaving the
    caller's single value as the only one present."""
    app = make_wrapper(
        FakeTransport(),
        gateway=GatewayConfig(default_provider="p"),
        config=Config(
            gateway=GatewayConfig(default_provider="p"),
            packs={"audit": {"endpoint": "https://audit.test", "threshold": 10}},
        ),
    )
    seen = []

    @app.before
    def spy(ctx):
        seen.append(dict(ctx.config.packs["audit"]))

    asyncio.run(
        app.complete(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            config={"packs": {"audit": {"threshold": 1}}},
        )
    )
    assert seen == [{"endpoint": "https://audit.test", "threshold": 1}]
