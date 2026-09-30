"""Pack discovery: the `margAI.packs` entry point group.

The interesting behaviour is the failure policy. An entry point is discovered
from whatever is installed, so a broken third-party package must not take the
gateway down -- but a pack the operator named explicitly must fail loudly,
because naming it is a request.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from conftest import FakeTransport, make_wrapper

from margAI.config import GatewayConfig
from margAI.packs import PackFailure, PackRecord, install_all


class FakeEntryPoint:
    """Stands in for an ``importlib.metadata`` entry point.

    Built here rather than monkeypatching ``entry_points`` so the test drives
    the real ``load()`` path, including the "entry point is not callable"
    failure that would otherwise only show up in a user's install. It is not
    an ``EntryPoint`` -- that is a concrete class, not a protocol -- so
    ``install_all`` takes a protocol instead.
    """

    name: str
    value: str

    def __init__(self, name: str, target: str, impl: Any = None, *, raises: bool = False) -> None:
        self.name = name
        self.value = target
        self._impl = impl
        self._raises = raises

    def load(self) -> Any:
        if self._raises:
            raise ImportError(f"cannot import {self.value}")
        if self._impl is not None:
            return self._impl
        return lambda app: None


def pack_that_registers(app, namespace: str, tag: str) -> None:
    """A realistically-shaped pack: claim the namespace, then a handler in it.

    Uses the same two steps a real pack does -- `install_bangtags` to claim the
    namespace, `app.pack(...)` so registrations are attributed -- because a
    test helper that skipped the pack context would register into `margAI` and
    prove nothing about namespacing.
    """
    from margAI import install_bangtags

    install_bangtags(app, namespace=namespace)
    with app.pack(f"test-{namespace}", namespace=namespace):

        @app.on(tag)
        def handle(ctx, tag):
            ctx.state[f"hit-{namespace}"] = True


def _none() -> PackFailure:
    """Stand-in so a test can read `.reason` without asserting on Optional.

    The point of the assertion that uses it is the *message*; the fact that a
    failure exists is asserted separately with `installed is False`.
    """
    return PackFailure(name="<none>", reason="")


def install(ep_name: str, **kwargs: Any):
    def _install(app):
        pack_that_registers(app, namespace=kwargs.get("ns", "ep"), tag=kwargs.get("tag", "ep-tag"))

    return FakeEntryPoint(ep_name, f"tests.test_packs:{ep_name}", _install)


# -- the happy path ---------------------------------------------------------


def test_a_discovered_pack_is_installed():
    app = make_wrapper(FakeTransport())
    records = install_all(app, eps=[install("demo")])
    assert [r.name for r in records] == ["demo"]
    assert records[0].installed is True


def test_a_discovered_pack_claims_a_working_tag():
    """Installed-yes only proves a callable returned. The tag diff is what
    proves it registered the thing the operator expected."""
    app = make_wrapper(FakeTransport())
    records = install_all(app, eps=[install("demo", tag="audit")])
    assert records[0].tags == ("ep:audit",)
    box: list[Any] = []

    @app.before(order=-40)
    def capture(ctx):
        box.append(ctx)

    body = {"model": "margAI/openai/gpt-4o", "messages": [{"role": "user", "content": "!ep: audit\nhi"}]}
    assert asyncio.run(app.complete(body)).status == 200
    assert box[0].state["hit-ep"] is True


def test_packs_install_in_a_stable_order():
    app = make_wrapper(FakeTransport())
    eps = [install("zebra"), install("alpha"), install("mango")]
    records = install_all(app, eps=eps)
    assert [r.name for r in records] == ["alpha", "mango", "zebra"]


def test_two_packs_in_different_namespaces_both_work():
    """One prompt, two namespaces, two handlers -- the coexistence case that
    the whole namespace design exists for."""
    app = make_wrapper(FakeTransport())
    install_all(app, eps=[install("a", ns="one", tag="x"), install("b", ns="two", tag="y")])
    box: list[Any] = []

    @app.before(order=-40)
    def capture(ctx):
        box.append(ctx)

    prompt = "!one: x !two: y\nReview Acme Ltd."
    body = {"model": "margAI/openai/gpt-4o", "messages": [{"role": "user", "content": prompt}]}
    assert asyncio.run(app.complete(body)).status == 200
    assert box[0].state["hit-one"] is True
    assert box[0].state["hit-two"] is True
    # And neither pack stole the other's text.
    sent = app.transport.requested[-1]  # type: ignore[attr-defined]
    assert sent.json["messages"][0]["content"] == "Review Acme Ltd."


# -- failure policy ---------------------------------------------------------


def test_a_broken_discovered_pack_does_not_stop_the_gateway(caplog):
    """Entry points come from whatever is installed. An unrelated broken
    dependency must not take production down with a traceback about someone
    else's code."""
    app = make_wrapper(FakeTransport())
    broken = FakeEntryPoint("broken", "nope:install", raises=True)
    records = install_all(app, eps=[broken, install("good")])
    by_name = {r.name: r for r in records}
    assert by_name["good"].installed is True
    assert by_name["broken"].installed is False
    assert "ImportError" in (by_name["broken"].failure or _none()).reason


def test_a_broken_pack_is_reported_not_swallowed(caplog):
    app = make_wrapper(FakeTransport())
    with caplog.at_level("ERROR"):
        install_all(app, eps=[FakeEntryPoint("broken", "nope:install", raises=True)])
    assert "broken" in caplog.text
    assert "cannot import" in caplog.text


def test_an_entry_point_that_is_not_callable_is_a_failure_not_a_crash():
    app = make_wrapper(FakeTransport())
    not_callable = FakeEntryPoint("weird", "weird:thing", impl="just a string")
    (record,) = install_all(app, eps=[not_callable])
    assert record.installed is False
    assert "callable" in (record.failure or _none()).reason


def test_a_named_pack_that_fails_raises():
    """Naming a pack is a request. Answering it with a log line is how "I
    enabled the pack and nothing happened" happens."""
    app = make_wrapper(FakeTransport())
    with pytest.raises(ImportError):
        install_all(app, eps=[FakeEntryPoint("broken", "nope:install", raises=True)], only=["broken"])


def test_only_restricts_to_the_named_packs():
    app = make_wrapper(FakeTransport())
    records = install_all(app, eps=[install("a"), install("b")], only=["b"])
    assert [r.name for r in records] == ["b"]


# -- opting out -------------------------------------------------------------


def test_a_skipped_pack_is_not_installed_and_does_not_fail():
    app = make_wrapper(FakeTransport())
    records = install_all(app, eps=[install("a")], skip=["a"])
    assert records == []


def test_discovery_can_be_switched_off_entirely():
    app = make_wrapper(FakeTransport())
    assert install_all(app, eps=[install("a")], enabled=False) == []


def test_a_pack_disabled_in_config_is_skipped():
    from conftest import make_config

    config = make_config(packs={"quiet": {"enabled": False}})
    app = make_wrapper(FakeTransport(), config=config)
    records = install_all(app, eps=[install("quiet")])
    assert records == []


def test_a_pack_marked_enabled_in_config_is_installed():
    from conftest import make_config

    app = make_wrapper(FakeTransport(), config=make_config(packs={"loud": {"enabled": True}}))
    records = install_all(app, eps=[install("loud")])
    assert [r.name for r in records] == ["loud"]


# -- through the wrapper ----------------------------------------------------


def test_the_wrapper_installs_discovered_packs(monkeypatch):
    from margAI import wrapper as wrapper_mod

    records = [PackRecord(name="demo", target="x", installed=True, tags=("ep:audit",))]
    monkeypatch.setattr(wrapper_mod, "install_all", lambda app, **kw: records)
    app = make_wrapper(FakeTransport())
    assert [r.name for r in app.packs] == ["demo"]
    assert app.pack_failures == []


def test_the_wrapper_surfaces_discovery_failures(monkeypatch):
    from margAI import wrapper as wrapper_mod

    failure = PackFailure(name="broken", reason="ImportError: nope")
    monkeypatch.setattr(
        wrapper_mod,
        "install_all",
        lambda app, **kw: [PackRecord(name="broken", target="x", installed=False, failure=failure)],
    )
    app = make_wrapper(FakeTransport())
    assert app.pack_failures == [failure]


def test_discovery_is_off_when_the_gateway_says_so(monkeypatch):
    seen: dict[str, Any] = {}

    def fake(app, **kwargs):
        seen.update(kwargs)
        return []

    from margAI import wrapper as wrapper_mod

    monkeypatch.setattr(wrapper_mod, "install_all", fake)

    from margAI.config import Config
    from margAI.providers import build_providers

    config = Config(gateway=GatewayConfig(pack_discovery=False))
    from margAI import Wrapper

    Wrapper(
        providers=build_providers(config),
        transport=FakeTransport(),
        config=config,
    )
    assert seen["enabled"] is False
    assert seen["only"] is None


def test_packs_only_is_passed_through(monkeypatch):

    from margAI import Wrapper
    from margAI import wrapper as wrapper_mod
    from margAI.config import Config
    from margAI.providers import build_providers

    seen: dict[str, Any] = {}
    monkeypatch.setattr(wrapper_mod, "install_all", lambda app, **kw: (seen.update(kw), [])[1])
    config = Config(gateway=GatewayConfig(packs_only=("only-me",)))
    Wrapper(providers=build_providers(config), transport=FakeTransport(), config=config)
    assert seen["only"] == ("only-me",)


# -- the built-in pack -------------------------------------------------------


def test_the_builtin_pack_is_installed_zero_config():
    """It ships inside this distribution, so the entry point group cannot report
    it -- but it still has to install, or `!margAI: route=` is silently dead."""
    from margAI import Wrapper
    from margAI.config import Config, GatewayConfig
    from margAI.providers import build_providers

    config = Config(gateway=GatewayConfig())
    wrapper = Wrapper(providers=build_providers(config), transport=FakeTransport(), config=config)
    assert {"route", "provider", "model", "not", "only", "cost", "think"} <= {
        name.rpartition(":")[2] for name in wrapper.tags.known()
    }


def test_the_builtin_pack_is_recorded_so_the_doctor_can_report_it():
    """Installed through the same machinery as a third-party pack, so it gets a
    record: a report built from config alone would describe tags nobody
    registered."""
    from margAI import Wrapper
    from margAI.config import Config, GatewayConfig
    from margAI.providers import build_providers

    config = Config(gateway=GatewayConfig())
    wrapper = Wrapper(providers=build_providers(config), transport=FakeTransport(), config=config)
    (record,) = wrapper.packs
    assert record.name == "margAI.builtin_tags"
    assert record.installed
    assert "margAI:route" in record.tags


def test_the_builtin_pack_can_be_disabled_by_name():
    """Opting out must be a table an operator can write, not a code change."""
    from margAI import Wrapper
    from margAI.config import Config, GatewayConfig
    from margAI.providers import build_providers

    config = Config(
        gateway=GatewayConfig(),
        packs={"margAI.builtin_tags": {"enabled": False}},
    )
    wrapper = Wrapper(providers=build_providers(config), transport=FakeTransport(), config=config)
    assert wrapper.tags.known() == set()
    assert wrapper.packs == []


def test_pack_discovery_does_not_disable_the_builtin_pack():
    """`pack_discovery` governs scanning for *third-party* packs. Letting it
    take away `route=` and `provider=` -- which the documentation calls
    zero-config -- would make one flag silently rewrite the language."""
    from margAI import Wrapper
    from margAI.config import Config, GatewayConfig
    from margAI.providers import build_providers

    config = Config(gateway=GatewayConfig(pack_discovery=False))
    wrapper = Wrapper(providers=build_providers(config), transport=FakeTransport(), config=config)
    assert "margai:route" in wrapper.tags.known()


def test_packs_only_can_exclude_the_builtin_pack():
    from margAI import Wrapper
    from margAI.config import Config, GatewayConfig
    from margAI.providers import build_providers

    config = Config(gateway=GatewayConfig(packs_only=("something-else",)))
    wrapper = Wrapper(providers=build_providers(config), transport=FakeTransport(), config=config)
    assert wrapper.tags.known() == set()


def test_gateway_load_builtin_tags_is_a_deprecated_no_op():
    """It used to be the opt-in. Installing the pack a second time made every
    built-in directive log "replacing before handler" for itself."""
    from margAI import Gateway
    from margAI.config import Config, GatewayConfig

    with pytest.warns(DeprecationWarning, match="no-op"):
        app = Gateway(Config(gateway=GatewayConfig()), load_builtin_tags=True)
    assert [r.name for r in app.wrapper.packs] == ["margAI.builtin_tags"]
