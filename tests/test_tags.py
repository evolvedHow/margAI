"""The tag system: parsing, ownership, ordering, and the no-hardcoding rule.

The last test in this file is the one that matters most. Every other test
describes how the pieces work; that one makes it impossible to add a tag
without going through the mechanism, which is what keeps the pieces working.
"""

from __future__ import annotations

import ast
import asyncio
import logging
import pathlib
from typing import Any

import pytest
from conftest import FakeTransport, UpstreamResponse, chat_payload, make_wrapper

from margAI import Wrapper, install_bangtags
from margAI.bangtag import Directive, Tag, find_directive, find_directives, strip_directives
from margAI.config import GatewayConfig
from margAI.core.context import RequestContext
from margAI.core.errors import ApiError
from margAI.core.hooks import HookKind
from margAI.core.tags import (
    RESERVED_NAMESPACE,
    ROOT_PACK,
    TagConflictError,
    TagRegistry,
    namespace_key,
    qualified_key,
    qualify,
)

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "margAI"

# Orchestration modules. None of them may name a tag: they parse, dispatch and
# route, and a `if name == "route"` in any of them is a bug by definition.
ORCHESTRATION = (
    "bangtag.py",
    "wrapper.py",
    "routing.py",
    "core/events.py",
    "core/tags.py",
    "core/intent.py",
    "core/context.py",
    "core/hooks.py",
    "core/marglets.py",
    "core/router.py",
)


def new_wrapper(**gateway: Any) -> Wrapper:
    """A gateway with a real config, so `on_unknown_tag` is settable."""
    return make_wrapper(FakeTransport(), gateway=GatewayConfig(**gateway))


def fresh_ctx() -> RequestContext:
    """A real request context. A hand-rolled stand-in would let a handler pass
    by touching only the attributes it happens to use."""
    return RequestContext(body={}, state={})


def call(app: Any, *tags: Tag | str, directive_ns: str = "margAI", model: str = "margAI/openai/gpt-4o") -> Any:
    """Make a real call carrying ``tags`` in a bangtag directive.

    ``directive_ns`` is a convenience for the single-namespace case; a test
    that needs two namespaces on one line passes a bare string as the first
    tag and leaves the namespace implicit.
    """
    rendered = " ".join(str(t) for t in tags)
    directive = rendered if rendered.startswith("!") else f"!{directive_ns}: {rendered}"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": f"{directive}\nhello"}],
    }
    return asyncio.run(app.complete(body))


def ask(app: Any, *tags: Tag | str, **kwargs: Any) -> RequestContext:
    """Make a successful call and hand back the context it ran with.

    The built-in tags are `before` handlers, so exercising them through
    `run_before` would test the registry rather than the gateway; going
    through `complete` is what proves the wiring, the parse and the handler
    all line up.
    """
    captured: list[RequestContext] = []

    @app.before(order=-40)
    def capture(ctx: RequestContext) -> None:
        captured.append(ctx)

    result = call(app, *tags, **kwargs)
    assert result.status == 200, result.body
    return captured[-1]


# -- parsing ---------------------------------------------------------------


def test_a_leading_block_is_the_canonical_form():
    text = "!margAI: think\n\nWhat is the capital of France?"
    found = find_directives(text)
    assert [d.namespace for d in found] == ["margAI"]
    assert [t.name for t in found[0].tags] == ["think"]
    assert found[0].leading is True
    assert strip_directives(text, found[0].spans) == "What is the capital of France?"


def test_blank_lines_before_the_block_do_not_cost_the_prompt():
    text = "\n\n!sc: audit\n\nReview this contract."
    found = find_directives(text)
    assert found[0].tags[0].name == "audit"
    assert strip_directives(text, found[0].spans) == "Review this contract."


def test_an_inline_directive_still_parses_but_is_not_leading():
    """The documented alternative form. It parses, and the directive is removed,
    but the tag is marked `leading=False` so a handler can tell the difference
    between a deliberate header and a mention in passing prose."""
    text = "make it terse  !margAI: refine"
    found = find_directives(text)
    assert found[0].leading is False
    assert [t.name for t in found[0].tags] == ["refine"]
    assert strip_directives(text, found[0].spans) == "make it terse"


def test_several_namespaces_on_one_line_split_cleanly():
    """The bug that started this: `!margAI:` used to swallow the directive after
    it, so the second namespace's tags were parsed as the first's values."""
    found = find_directives("!hr: approve !margAI: think")
    assert {d.namespace: [t.name for t in d.tags] for d in found} == {
        "hr": ["approve"],
        "margAI": ["think"],
    }


def test_every_namespace_is_found_not_just_the_first():
    found = find_directives("!margAI: think !sc: audit region=eu !hr: approve")
    assert [d.namespace for d in found] == ["margAI", "sc", "hr"]


def test_repeating_a_namespace_merges_it_without_losing_either():
    found = find_directives("!margAI: think\n!sc: audit\n!margAI: route=local")
    margai = next(d for d in found if d.namespace == "margAI")
    assert [t.name for t in margai.tags] == ["think", "route"]
    assert margai.leading is True


def test_a_repeated_directive_is_stripped_in_full():
    """Both occurrences have to go, and removing the first must not shift the
    text the second was located in."""
    text = "!margAI: think\n!margAI: route=local\nTell me a story."
    found = find_directives(text)
    spans = tuple(s for d in found for s in d.spans)
    assert strip_directives(text, spans) == "Tell me a story."


def test_values_and_bare_tags():
    tags = {t.name: t.value for t in find_directives("!margAI: cheap, cost=0.5")[0].tags}
    assert tags == {"cheap": None, "cost": "0.5"}


def test_two_packs_stripping_one_line_do_not_glue_the_prompt_on():
    """The regression that made "Review Acme Ltd." into three tags.

    Each pack parses and strips only its own namespace, and it re-reads the
    text the previous hook left behind. When both directives share a line, the
    first hook to run must not eat the newline -- doing so welds the remaining
    prompt onto the surviving directive, and the next hook reads the whole
    sentence as that namespace's tags.
    """
    text = "!sc: audit region=eu !margAI: think=high\nReview Acme Ltd."

    # margAI's hook runs first and removes its mid-line span.
    after_margai = strip_directives(text, ("!margAI: think=high",))
    assert after_margai == "!sc: audit region=eu \nReview Acme Ltd."
    assert [str(t) for t in find_directives(after_margai)[0].tags] == ["audit", "region=eu"]

    # sc's hook then reads only its own directive, not the prompt.
    sc = find_directives(after_margai)[0]
    assert strip_directives(after_margai, sc.spans) == "Review Acme Ltd."


def test_a_directive_alone_on_a_line_takes_the_whole_line():
    """The other half of the rule: leaving the newline behind would put a blank
    line where the directive was."""
    text = "Do this\n!margAI: refine\nDo that"
    found = find_directives(text)
    assert strip_directives(text, found[0].spans) == "Do this\nDo that"


def test_two_installed_namespaces_contribute_to_one_tag_list():
    """The regression behind a much quieter bug.

    Every installed namespace gets its own request hook, and each one parses
    only its own directive. They share one tag list, so each must add to it
    rather than replace it -- otherwise the last hook to run throws away
    everyone else's tags, and which pack works at all depends on registration
    order. Here the framework's own `think` and a domain pack's `audit` are on
    the same line, and both have to take effect.
    """
    app = new_wrapper()
    install_bangtags(app, namespace="sc")
    with app.pack("supply-chain", namespace="sc"):
        app.on("audit", order=-20)(lambda ctx, tag: ctx.add_system_prompt("audit it"))

    seen = ask(app, "!sc: audit !margAI: think=high")
    tags = {getattr(t, "qualified", None) for t in seen.tags}
    assert tags == {"sc:audit", "margai:think"}
    assert seen.body["reasoning_effort"] == "high"
    assert seen.body["messages"][0]["content"] == "audit it"


def test_a_pack_installed_after_the_builtin_pack_still_works():
    """Same thing, in the other registration order -- the point is that the
    outcome does not depend on it."""
    app = new_wrapper()
    with app.pack("hr", namespace="hr"):
        app.on("approve")(lambda ctx, tag: ctx.add_system_prompt("approved"))
    install_bangtags(app, namespace="hr")

    seen = ask(app, "!hr: approve !margAI: route=openai/gpt-4o")
    # Membership, not order: each parser only sees its own directive, so the
    # order tags land in is parser-registration order. Both effects are what
    # this test is for -- neither pack may be lost to the other.
    assert {getattr(t, "qualified", None) for t in seen.tags} == {"hr:approve", "margai:route"}
    assert seen.body["messages"][0]["content"] == "approved"
    assert seen.intent.provider == "openai"


# -- the allowlist: the trust boundary for a message margAI did not write ----
#
# Everything above treats a directive as something the operator wrote. These
# tests are the other case -- a user wrote it -- and the one that matters most
# is the last: a denied tag must not survive in the prompt either.


def test_the_allowlist_defaults_to_off_so_the_feature_is_unchanged():
    """No `allow` means the message is trusted, which is the documented default.

    Asserted explicitly because it is the compatibility contract: an allowlist
    that defaulted to closed would silently stop every existing deployment's
    `route=` from working, which is a louder failure but a worse design -- the
    layer is opt-in, and opting in is where the trust decision belongs.
    """
    app = new_wrapper()
    seen = ask(app, "route=openai/gpt-4o-mini")
    assert seen.intent.model == "gpt-4o-mini"


def test_a_tag_off_the_allowlist_cannot_pin_the_route():
    """The actual attack: `!margAI: route=` is a spend control, so it is denied.

    This also covers the ordering: the built-in tag pack has already installed
    the `margAI` namespace by the time the test calls `install_bangtags`, so
    the allowlist here is a *tightening* of an installed parser. If the policy
    were only read at first install, this would pass as a no-op and assert
    nothing.
    """
    app = new_wrapper()
    install_bangtags(app, allow=["think"])
    seen = ask(app, "route=openai/gpt-4o-mini")
    assert seen.intent.model is None


def test_a_repeat_install_without_an_allowlist_does_not_reopen_it():
    """Tightening is one-way. A second plain `install_bangtags(app)` is far
    more likely to be "I forgot this was already set up" than a deliberate
    unlock, and it must not hand `route=` back."""
    app = new_wrapper()
    install_bangtags(app, allow=["think"])
    install_bangtags(app)

    seen = ask(app, "route=openai/gpt-4o-mini")
    assert seen.intent.model is None


def test_an_allowlisted_tag_still_works():
    """A closed door next to the one that matters: the allowlist is a filter,
    not a switch. An allowlist of `think` has to leave `think` working."""
    app = new_wrapper()
    install_bangtags(app, allow=["think"])
    seen = ask(app, "think=high")
    assert seen.body["reasoning_effort"] == "high"


def test_an_allowlist_admits_a_namespaced_name():
    """`margAI:think` is accepted as well as `think`, so a deployment that
    installs two namespaces can say which one it means."""
    app = new_wrapper()
    install_bangtags(app, allow=["margAI:think"])
    seen = ask(app, "think=high")
    assert seen.body["reasoning_effort"] == "high"


def test_a_denied_directive_is_still_stripped_from_the_prompt():
    """The regression worth having.

    When every tag on a directive is denied, the natural implementation is to
    drop the whole directive -- and with it the strip. The caller then gets
    their own `!margAI: route=openai/gpt-4o` handed to the model as prose, and
    an instruction-following model is exactly the thing that will act on it.
    Denying a tag has to take the text with it.
    """
    app = new_wrapper()
    install_bangtags(app, allow=["think"])
    seen = ask(app, "route=openai/gpt-4o")
    assert seen.body["messages"][0]["content"] == "hello"


def test_an_empty_allowlist_admits_nothing_but_still_strips():
    """`allow=()` is the way to keep the parsing without letting anything
    through -- for a deployment that strips directives out of prompts but
    routes on the model field alone."""
    app = new_wrapper()
    install_bangtags(app, allow=[])
    seen = ask(app, "think=high")
    assert seen.body.get("reasoning_effort") is None
    assert seen.body["messages"][0]["content"] == "hello"


def test_a_dropped_tag_is_reported(caplog: pytest.LogCaptureFixture):
    """A tag the operator configured away, set from a message, is either a
    misconfiguration or an attempt. Both belong in a log the operator reads,
    so this is a warning and not a debug."""
    app = new_wrapper()
    install_bangtags(app, allow=["think"])
    with caplog.at_level(logging.WARNING, logger="margAI.bangtag"):
        ask(app, "route=openai/gpt-4o")
    assert "route" in caplog.text


def test_an_allowlist_only_governs_its_own_namespace():
    """Two installed namespaces, two allowlists. The framework's must not
    start vetting a pack's tags, or a pack would need its operator to restate
    the framework's list."""
    app = new_wrapper()
    install_bangtags(app, allow=["think"])
    install_bangtags(app, namespace="sc", allow=["audit"])
    with app.pack("supply-chain", namespace="sc"):
        app.on("audit", order=-20)(lambda ctx, tag: ctx.add_system_prompt("audit it"))

    seen = ask(app, "!sc: audit !margAI: route=openai/gpt-4o-mini")
    assert {getattr(t, "qualified", None) for t in seen.tags} == {"sc:audit"}
    assert seen.intent.model is None


def test_set_tags_with_a_namespace_replaces_only_its_own():
    """The contract the two parsers rely on, stated on its own."""
    from margAI.core.context import RequestContext

    context = RequestContext(body={}, state={})
    context.set_tags([Tag("a", namespace="one")], namespace="one")
    context.set_tags([Tag("b", namespace="two")], namespace="two")
    context.set_tags([Tag("a2", namespace="one")], namespace="one")
    # Kept tags keep their position; the namespace being replaced is dropped
    # and re-added, so a re-run cannot reorder the call's tags.
    assert [str(t) for t in context.tags] == ["b", "a2"]
    # Without a namespace it is a wholesale reset, as it always was.
    context.set_tags([Tag("solo")])
    assert [str(t) for t in context.tags] == ["solo"]


def test_a_prompt_with_no_directive_yields_nothing():
    assert find_directives("just a question") == []
    assert find_directives("") == []


def test_find_directive_keeps_the_single_namespace_view():
    text, tags = find_directive("!margAI: cheap !sc: audit")
    assert text == "!margAI: cheap"
    assert [t.name for t in tags] == ["cheap"]
    assert find_directive("!sc: audit", "margAI") == (None, [])


# -- identity --------------------------------------------------------------


def test_namespace_identity_folds_case_but_display_does_not():
    """`!MARGAI:` is the same tag as `!margAI:`, and introspection still spells
    the brand the way a pack registered it."""
    registry = TagRegistry()
    registry.register(name="think", namespace=RESERVED_NAMESPACE, phase="before", fn=lambda ctx, tag: None)
    assert "margai:think" in registry.known()
    assert registry.describe()["margAI"][0]["name"] == "think"
    assert namespace_key("  MARGAI ") == namespace_key(RESERVED_NAMESPACE)
    assert qualified_key("think", "MARGAI") == qualify(Tag("think", namespace="margAI"))


def test_qualify_degrades_for_a_foreign_tag():
    """The pipeline only requires `.name`; a hand-built object from elsewhere
    must not raise."""
    class Foreign:
        name = "shout"

    assert qualify(Foreign()) == "shout"
    assert qualify(object()) == ""


# -- packs and ownership ---------------------------------------------------


def test_a_pack_owns_its_namespace():
    app = new_wrapper()
    with app.pack("supply-chain", namespace="sc"):
        app.on("audit")(lambda ctx, tag: None)
    assert app.tags.owner("sc") == "supply-chain"
    assert "sc:audit" in app.tags.known()


def test_two_packs_cannot_claim_one_namespace():
    app = new_wrapper()
    with app.pack("supply-chain", namespace="sc"):
        app.on("audit")(lambda ctx, tag: None)
    with pytest.raises(TagConflictError, match="owned by pack 'supply-chain'"), app.pack("hr", namespace="sc"):
        pass


def test_a_pack_cannot_claim_the_reserved_namespace():
    app = new_wrapper()
    with pytest.raises(TagConflictError, match="owned by pack"), app.pack(
        "rogue", namespace=RESERVED_NAMESPACE
    ):
        pass


def test_a_pack_cannot_register_into_the_reserved_namespace():
    """Namespace ownership and the reserved check are separate guards: a pack
    that never calls `pack()` still may not reach in."""
    app = new_wrapper()
    with app.pack("rogue", namespace="rogue-ns"), pytest.raises(
        TagConflictError, match="reserved for the framework"
    ):
        app.on("sneaky", namespace=RESERVED_NAMESPACE)(lambda ctx, tag: None)


def test_a_pack_cannot_replace_another_packs_handler():
    """The namespace guard fires first, which is stronger: a pack cannot even
    *name* another pack's directive, so there is nothing left to guard."""
    app = new_wrapper()
    with app.pack("supply-chain", namespace="sc"):
        app.on("audit")(lambda ctx, tag: None)
    with app.pack("hr", namespace="hr-ns"), pytest.raises(
        TagConflictError, match="namespace 'sc' is owned by pack"
    ):
        app.on("audit", namespace="sc", replace=True)(lambda ctx, tag: None)
    # ...and without `replace` it is still refused, the ordinary way.
    with app.pack("hr2", namespace="hr2-ns"), pytest.raises(TagConflictError, match="may not claim it"):
        app.on("audit", namespace="sc")(lambda ctx, tag: None)


def test_a_pack_may_add_to_its_own_tag():
    """Accumulating handlers is composition, not overwriting: the whole point
    of `!margAI: think` steering routing *and* setting reasoning effort."""
    app = new_wrapper()
    with app.pack("sc", namespace="sc"):
        app.on("audit")(lambda ctx, tag: None)
        app.on("audit", order=5)(lambda ctx, tag: None)
    assert len(app.tags.handlers("sc:audit", "before")) == 2


def test_app_code_may_displace_a_pack_handler_on_purpose():
    app = new_wrapper()
    with app.pack("supply-chain", namespace="sc"):
        app.on("audit")(lambda ctx, tag: None)
    app.on("audit", namespace="sc", replace=True)(lambda ctx, tag: None)
    handlers = app.tags.handlers("sc:audit", "before")
    assert len(handlers) == 1
    assert handlers[0].pack == ROOT_PACK


def test_a_pack_reloading_its_own_namespace_is_not_a_conflict():
    app = new_wrapper()
    for _ in range(2):
        with app.pack("supply-chain", namespace="sc"):
            app.on("audit", replace=True)(lambda ctx, tag: None)
    assert len(app.tags.handlers("sc:audit", "before")) == 1


def test_the_builtin_pack_is_a_guest_in_the_applications_namespace():
    """Installing margAI's own tags must not cost the application its own
    namespace -- otherwise `!margAI:` would be unusable by the very code that
    ships it, and the built-ins would own it instead of the app."""
    app = new_wrapper()
    assert app.tags.owner(RESERVED_NAMESPACE) == ROOT_PACK
    app.on("shout")(lambda ctx, tag: None)
    assert app.tags.owner(RESERVED_NAMESPACE) == ROOT_PACK
    assert "margai:shout" in app.tags.known()


# -- ordering --------------------------------------------------------------


def test_handlers_for_one_tag_run_in_order_then_registration_order():
    app = new_wrapper()
    seen: list[str] = []
    # `probe`, not `think`: this measures ordering, and `think` already has a
    # built-in handler that would join the run.
    app.on("probe", order=10)(lambda ctx, tag: seen.append("late"))
    app.on("probe", order=-10)(lambda ctx, tag: seen.append("early"))
    app.on("probe")(lambda ctx, tag: seen.append("default-1"))
    app.on("probe")(lambda ctx, tag: seen.append("default-2"))
    asyncio.run(app._events.run_before([Tag("probe")], fresh_ctx()))
    assert seen == ["early", "default-1", "default-2", "late"]


def test_a_handler_receives_the_tag_the_caller_wrote():
    """Several tags can share a handler; it gets the concrete tag, not the
    registry's canonical key."""
    app = new_wrapper()
    got: list[Any] = []
    with app.pack("sc", namespace="sc"):
        app.on("shared")(lambda ctx, tag: got.append(tag))
    asyncio.run(app._events.run_before([Tag("shared", namespace="sc", value="eu")], fresh_ctx()))
    assert got[0].value == "eu"
    assert got[0].namespace == "sc"


# -- introspection ---------------------------------------------------------


def test_describe_reports_namespace_phase_pack_and_doc():
    app = new_wrapper()
    with app.pack("supply-chain", namespace="sc", doc="Supply chain review"):
        @app.on("audit")
        def audit(ctx, tag):
            """Check a supplier against the approved list."""

    described = app.tags.describe()["sc"][0]
    assert described["name"] == "audit"
    assert described["phases"] == ["before"]
    assert described["packs"] == ["supply-chain"]
    assert described["doc"] == "Check a supplier against the approved list."


def test_docs_is_a_bare_name_view():
    registry = TagRegistry()
    registry.register(name="shout", namespace="sc", phase="before", fn=lambda ctx, tag: None)
    assert registry.docs() == {"shout": ""}
    assert registry.bare_names() == {"shout"}
    assert registry.owner_of_tag("sc:shout") == ROOT_PACK


# -- end to end ------------------------------------------------------------


def test_pack_directives_only_reach_their_own_handlers():
    """Two packs defining `approve` never meet: the namespace is what keeps
    them apart, so neither handler has to check who is calling."""
    app = new_wrapper()
    fired: list[str] = []
    with app.pack("supply-chain", namespace="sc"):
        app.on("approve")(lambda ctx, tag: fired.append("sc"))
    with app.pack("hr", namespace="hr"):
        app.on("approve")(lambda ctx, tag: fired.append("hr"))

    asyncio.run(app._events.run_before([Tag("approve", namespace="hr")], fresh_ctx()))
    assert fired == ["hr"]


def test_builtin_tags_are_installed_and_zero_config():
    """`!margAI: route=` is documented, so it has to work on a gateway nobody
    configured -- because a built-in pack is installed exactly like any other."""
    app = new_wrapper()
    assert {"route", "provider", "model", "not", "only", "cost", "think"} <= app.tags.bare_names()
    assert app.tags.owner(RESERVED_NAMESPACE) == ROOT_PACK


def test_think_sets_a_routing_neutral_request_parameter():
    """The two axes, kept apart: `think` shapes the body and says nothing to
    routing, so the intent is untouched."""
    app = new_wrapper()
    seen = ask(app, Tag("think"), Tag("think", value="high"))
    assert seen.body["reasoning_effort"] == "high"
    assert seen.intent.model is None
    assert seen.intent.provider is None


def test_a_bare_think_is_the_default_effort():
    assert ask(new_wrapper(), Tag("think")).body["reasoning_effort"] == "medium"


def test_think_rejects_an_unknown_effort():
    """A bad selector value is the caller's mistake, so it must surface as a 400
    naming the tag -- not as an unhandled 500."""
    result = call(new_wrapper(), Tag("think", value="extreme"))
    assert result.status == 400
    assert "think= wants one of" in result.body["error"]["message"]
    assert result.body["error"]["param"] == "bangtags"


def test_cost_prices_the_selector_chain():
    assert ask(new_wrapper(), Tag("cost", value="0.25")).intent.max_cost_per_1m == 0.25


def test_cost_rejects_a_non_number():
    result = call(new_wrapper(), Tag("cost", value="cheap"))
    assert result.status == 400
    assert "cost= wants a number" in result.body["error"]["message"]


def test_selectors_fill_the_intent_through_the_pack():
    seen = ask(new_wrapper(), Tag("route", value="local/qwen3"), Tag("not", value="openai"))
    assert seen.intent.provider == "local"
    assert seen.intent.model == "qwen3"
    assert seen.intent.exclude == frozenset({"openai"})


def test_a_more_specific_selector_wins_over_a_coarser_one():
    """`route=local/qwen3 model=qwen3-32b` -- `model` is applied later, so the
    narrower of the two claims is the one that survives."""
    seen = ask(new_wrapper(), Tag("route", value="local/qwen3"), Tag("model", value="qwen3-32b"))
    assert seen.intent.provider == "local"
    assert seen.intent.model == "qwen3-32b"


def test_provider_pins_the_provider_and_leaves_the_model_to_routing():
    assert ask(new_wrapper(), Tag("provider", value="openai")).intent.provider == "openai"


def test_repeated_policy_selectors_accumulate():
    seen = ask(new_wrapper(), Tag("not", value="openai"), Tag("not", value="groq"))
    assert seen.intent.exclude == frozenset({"openai", "groq"})


def test_policy_is_applied_before_a_pin():
    """A pin that contradicts the policy is still a pin, and the router refuses
    it loudly. Both facts have to be on the intent before anything reads it,
    which is what the handler `order` in builtin_tags guarantees."""
    seen = ask(
        new_wrapper(),
        Tag("route", value="local/qwen3"),
        Tag("only", value="llama3"),
        Tag("not", value="openai"),
    )
    assert seen.intent.model == "qwen3"
    assert seen.intent.require == frozenset({"llama3"})
    assert seen.intent.allows("local", "qwen3") is False


def test_a_selector_name_in_an_unowned_namespace_does_nothing():
    """The behavioural half of the no-hardcoding rule, and the strongest one.

    `!rogue: route=local/qwen3` carries a tag name the framework documents,
    in a namespace nobody owns. If any orchestration module branched on
    `name == "route"`, this would still route. Because the built-in handler is
    registered as `margAI:route`, the foreign tag is inert -- which is the
    whole claim, tested rather than asserted.
    """
    seen = ask(
        new_wrapper(),
        Tag("route", value="local/qwen3"),
        Tag("not", value="openai"),
        directive_ns="rogue",
    )
    assert seen.intent.model is None
    assert seen.intent.provider is None
    assert seen.intent.exclude == frozenset()


# -- unknown tags ----------------------------------------------------------


def test_an_unknown_tag_warns_by_default(caplog):
    app = new_wrapper()
    with caplog.at_level("WARNING", logger="margAI.wrapper"):
        app._check_tags([Tag("nonexistent")])
    assert any("no handler claims" in r.message for r in caplog.records)


def test_an_unknown_tag_can_be_an_error():
    app = new_wrapper(on_unknown_tag="error")
    with pytest.raises(ApiError) as excinfo:
        app._check_tags([Tag("nonexistent")])
    assert excinfo.value.status == 400
    assert "unknown bangtag" in excinfo.value.message


def test_an_unknown_tag_can_be_ignored(caplog):
    app = new_wrapper(on_unknown_tag="ignore")
    with caplog.at_level("WARNING", logger="margAI.wrapper"):
        app._check_tags([Tag("nonexistent")])
    assert caplog.records == []


def test_an_unrecognised_policy_falls_back_to_warning(caplog):
    """A typo in the config should not silently disable the check."""
    app = new_wrapper(on_unknown_tag="shout")
    with caplog.at_level("WARNING", logger="margAI.wrapper"):
        app._check_tags([Tag("nonexistent")])
    assert any("is not a known policy" in r.message for r in caplog.records)


def test_a_known_tag_is_never_reported(caplog):
    """Counted across every phase, so a tag handled only in `after` counts, and
    compared qualified, so `!hr: approve` is not an unknown `approve`."""
    app = new_wrapper()
    with app.pack("hr", namespace="hr"):
        app.on("approve", phase="after")(lambda payload, ctx, tag: payload)
    with caplog.at_level("WARNING", logger="margAI.wrapper"):
        app._check_tags([Tag("approve", namespace="hr"), Tag("route", value="x")])
    assert caplog.records == []


# -- the invariant ---------------------------------------------------------


def _compared_literals(tree: ast.AST) -> set[tuple[int, str]]:
    """Every string literal that is an *operand of a comparison*.

    That is the shape a hardcoded tag check takes: `name == "route"`, or
    `if tag.name in {"think", "plan"}`. Looking only at comparisons is what
    makes this check sound rather than noisy -- `"model"` and `"provider"`
    are also OpenAI request fields and router attributes, and they appear all
    over the orchestration layer as `param="model"` and `body["model"]`,
    neither of which is a comparison. Flagging those would make the rule
    impossible to keep, and a rule nobody can keep is not a rule.
    """
    out: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        for operand in operands:
            if isinstance(operand, ast.Constant) and isinstance(operand.value, str):
                out.add((operand.lineno, operand.value))
    return out


def test_no_orchestration_module_branches_on_a_bangtag_name():
    """The syntactic half of the rule: no orchestration module may decide
    behaviour by comparing against a tag name.

    Checked against the vocabulary a live gateway actually registers, so it
    cannot drift from the built-in pack, and it covers the namespace-owned
    selector names too -- a comparison on `"sc"` would be the same bug wearing
    a different hat.
    """
    app = new_wrapper()
    vocabulary = set(app.tags.known()) | app.tags.bare_names() | app.tags.namespaces()

    offenders: list[str] = []
    for filename in ORCHESTRATION:
        path = SRC / filename
        if not path.exists():
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for lineno, literal in sorted(_compared_literals(tree)):
            if literal in vocabulary:
                offenders.append(f"{filename}:{lineno}: {literal!r}")

    assert not offenders, (
        "orchestration code branches on a bangtag name:\n" + "\n".join(offenders)
    )


def test_every_builtin_tag_name_is_inert_in_a_foreign_namespace():
    """The behavioural half, and the one that really holds the design up.

    For every tag name the framework ships, the same name in a namespace
    nobody owns must do nothing at all. This is the no-hardcoding claim tested
    rather than asserted: if any orchestration module recognised a tag by
    name, this would be the test that failed, for that tag. It also cannot be
    satisfied by a well-meaning comment or a docstring -- only by the behaviour
    actually being absent.
    """
    app = new_wrapper()
    for name in sorted(app.tags.bare_names()):
        seen = ask(
            app,
            Tag(name, value="x", namespace="rogue"),
            Tag(name, namespace="rogue"),
            directive_ns="rogue",
        )
        assert seen.intent == seen.intent.__class__(), f"{name!r} reached the intent from a foreign namespace"
        assert "reasoning_effort" not in seen.body, f"{name!r} reached the body from a foreign namespace"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("!margAI: think\nhello", ["think"]),
        ("hello !margAI: think", ["think"]),
        ("!hr: a !margAI: b !sc: c", ["a", "b", "c"]),
        ("no tags here", []),
    ],
)
def test_tag_names_round_trip(text: str, expected: list[str]):
    found = find_directives(text)
    assert [t.name for d in found for t in d.tags] == expected


def test_directive_is_a_value_object():
    directive = Directive(namespace="sc", tags=(Tag("audit"),), spans=("!sc: audit",))
    assert directive.names == ("audit",)
    assert directive.text == "!sc: audit"
    with pytest.raises(AttributeError):
        directive.namespace = "hr"  # type: ignore[misc]


def test_a_fake_transport_gateway_still_answers():
    """Sanity: the built-in pack is inert unless a call actually uses it."""
    transport = FakeTransport(responses=[UpstreamResponse(200, chat_payload("hi"))])
    app = make_wrapper(transport)
    result = asyncio.run(
        app.complete({"model": "margAI/openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    )
    assert result.status == 200
    assert transport.requested[-1].json["messages"][0]["content"] == "hi"


def test_hook_accessors_do_not_hand_out_the_registrys_own_list():
    """`requests()` used to return the internal list while its three siblings
    returned copies, so `app._hooks.requests().clear()` silently unregistered
    every request hook on the gateway."""
    from conftest import FakeTransport, make_wrapper

    app = make_wrapper(FakeTransport())

    @app.before
    def keep(ctx):
        pass

    before = app._hooks.count(HookKind.REQUEST)
    assert before > 0

    for accessor in (app._hooks.requests, app._hooks.responses, app._hooks.streams, app._hooks.errors):
        accessor().clear()
    assert app._hooks.count(HookKind.REQUEST) == before
    assert app._hooks.count(HookKind.RESPONSE) > 0
