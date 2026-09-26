"""Router resolution and catalog tests."""

from __future__ import annotations

import pytest

from margAI.core import ApiError, ModelRouter
from margAI.core.router import Route

from conftest import make_config, provider_config
from margAI.providers import build_providers


def router(*profiles, expose="prefixed", default_provider=None):
    providers = build_providers(make_config(providers=tuple(profiles)))
    return ModelRouter(providers, prefix="margAI", expose=expose, default_provider=default_provider)


@pytest.fixture
def multi():
    return [
        provider_config(name="openai", models=("gpt-4o",)),
        provider_config(
            name="openrouter",
            base_url="https://openrouter.test/api/v1",
            models=("meta-llama/llama-3-70b", "gpt-4o"),
        ),
    ]


def test_resolve_namespaced(multi):
    r = router(*multi)
    assert r.resolve("margAI/openai/gpt-4o") == Route("openai", "gpt-4o", "qualified")
    assert r.resolve("margAI/openrouter/meta-llama/llama-3-70b") == Route(
        "openrouter", "meta-llama/llama-3-70b", "qualified"
    )


def test_default_provider_breaks_tie(multi):
    r = router(*multi, default_provider="openai")
    assert r.resolve("margAI/openai/gpt-4o") == Route("openai", "gpt-4o", "qualified")


def test_provider_default_model():
    r = router(
        provider_config(name="openai", models=("gpt-4o",), default_model="gpt-4o")
    )
    assert r.resolve("margAI/openai") == Route("openai", "gpt-4o", "provider_default")


def test_bare_model_ambiguous_without_default_provider(multi):
    """No default_provider to break the tie means the bare id is a client
    error, not a silent pick."""
    r = router(*multi)
    with pytest.raises(ApiError) as e:
        r.resolve("gpt-4o")
    assert e.value.status == 400
    assert "multiple providers" in e.value.message


def test_bare_model_accepted_regardless_of_expose(multi):
    """`expose` gates the catalog listing, never resolution."""
    r = router(*multi, expose="prefixed", default_provider="openai")
    assert r.resolve("gpt-4o") == Route("openai", "gpt-4o", "default_provider")


def test_unprefixed_raw_expose(multi):
    r = router(*multi, expose="both", default_provider="openai")
    assert r.resolve("gpt-4o") == Route("openai", "gpt-4o", "default_provider")


def test_unprefixed_unique_bare_model_resolves(multi):
    """Only one provider offers llama-3-70b, so no tie to break."""
    r = router(*multi, expose="prefixed")
    assert r.resolve("meta-llama/llama-3-70b") == Route("openrouter", "meta-llama/llama-3-70b", "bare")


def test_ambiguous_bare_model():
    r = router(
        provider_config(name="openai", models=("gpt-4o",)),
        provider_config(name="openrouter", base_url="x/v1", models=("gpt-4o",)),
        expose="both",
    )
    with pytest.raises(ApiError) as e:
        r.resolve("gpt-4o")  # exists on both openai and openrouter
    assert e.value.status == 400
    assert "multiple providers" in e.value.message


def test_try_resolve_returns_none_instead_of_raising(multi):
    """The dynamic path needs "cannot resolve" to be a value, not an
    exception, so it can fall through to the selector chain."""
    r = router(*multi)
    assert r.try_resolve("no-such-model") is None
    assert r.try_resolve("margAI/openai/gpt-4o") == Route("openai", "gpt-4o", "qualified")


def test_candidates_are_stable_and_sorted(multi):
    r = router(*multi)
    cands = r.candidates()
    assert [(c.provider, c.model) for c in cands] == [
        ("openai", "gpt-4o"),
        ("openrouter", "gpt-4o"),
        ("openrouter", "meta-llama/llama-3-70b"),
    ]
    assert r.candidates() == cands  # deterministic across calls


def test_dynamic_id_is_prefixed_and_reserved(multi):
    r = router(*multi)
    assert r.dynamic_id == "margAI/dynamic"
    assert r.is_dynamic("margAI/dynamic")
    assert r.is_dynamic("dynamic")  # bare form, nothing claims the name
    assert not r.is_dynamic("gpt-4o")
    assert not r.is_dynamic(None)


def test_bare_dynamic_name_yields_to_a_real_model(multi):
    """If some provider really offers a model called `dynamic`, the bare word
    is that model and only the prefixed id is the reserved one."""
    r = router(
        provider_config(name="openai", models=("gpt-4o", "dynamic")),
        provider_config(name="openrouter", base_url="x/v1", models=("dynamic",)),
        default_provider="openai",
    )
    assert r.is_dynamic("margAI/dynamic")
    assert not r.is_dynamic("dynamic")
    assert r.resolve("dynamic") == Route("openai", "dynamic", "default_provider")


def test_unknown_provider_hint(multi):
    func = router(*multi)
    with pytest.raises(ApiError) as e:
        func.resolve("margAI/notaprovider/gpt-4o")
    assert "notaprovider" in e.value.message


def test_catalog_includes_dynamic_when_asked(multi):
    r = router(*multi)
    assert "margAI/dynamic" not in [i["id"] for i in r.catalog({"openai": ["gpt-4o"]})]
    ids = [i["id"] for i in r.catalog({"openai": ["gpt-4o"]}, include_dynamic=True)]
    assert ids[0] == "margAI/dynamic"


def test_catalog_prefixed_only(multi):
    r = router(*multi)
    items = r.catalog({"openai": ["gpt-4o"], "openrouter": ["meta-llama/llama-3-70b"]})
    ids = [i["id"] for i in items]
    assert ids == ["margAI/openai/gpt-4o", "margAI/openrouter/meta-llama/llama-3-70b"]
    assert all(i["parent"] for i in items)


def test_catalog_both_expose(multi):
    r = router(*multi, expose="both")
    items = r.catalog({"openai": ["gpt-4o"]})
    ids = [i["id"] for i in items]
    assert ids == ["margAI/openai/gpt-4o", "gpt-4o"]


def test_no_model_specified(multi):
    r = router(*multi)
    with pytest.raises(ApiError) as e:
        r.resolve(None)
    assert e.value.status == 400