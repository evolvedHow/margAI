"""Router resolution and catalog tests."""

from __future__ import annotations

import pytest

from margai.core import ApiError, ModelRouter
from margai.core.router import Route

from conftest import make_config, provider_config
from margai.providers import build_providers


def router(*profiles, expose="prefixed", default_provider=None):
    providers = build_providers(make_config(providers=tuple(profiles)))
    return ModelRouter(providers, prefix="marg", expose=expose, default_provider=default_provider)


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
    assert r.resolve("marg/openai/gpt-4o") == Route("openai", "gpt-4o")
    assert r.resolve("marg/openrouter/meta-llama/llama-3-70b") == Route("openrouter", "meta-llama/llama-3-70b")


def test_default_provider_breaks_tie(multi):
    r = router(*multi, default_provider="openai")
    assert r.resolve("marg/openai/gpt-4o") == Route("openai", "gpt-4o")


def test_provider_default_model():
    r = router(
        provider_config(name="openai", models=("gpt-4o",), default_model="gpt-4o")
    )
    assert r.resolve("marg/openai").model == "gpt-4o"


def test_unprefixed_rejected_by_default(multi):
    r = router(*multi)
    with pytest.raises(ApiError) as e:
        r.resolve("gpt-4o")
    assert e.value.status == 404


def test_unprefixed_raw_expose(multi):
    r = router(*multi, expose="both", default_provider="openai")
    assert r.resolve("gpt-4o") == Route("openai", "gpt-4o")


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


def test_unknown_provider_hint(multi):
    func = router(*multi)
    with pytest.raises(ApiError) as e:
        func.resolve("marg/notaprovider/gpt-4o")
    assert "notaprovider" in e.value.message


def test_catalog_prefixed_only(multi):
    r = router(*multi)
    items = r.catalog({"openai": ["gpt-4o"], "openrouter": ["meta-llama/llama-3-70b"]})
    ids = [i["id"] for i in items]
    assert ids == ["marg/openai/gpt-4o", "marg/openrouter/meta-llama/llama-3-70b"]
    assert all(i["parent"] for i in items)


def test_catalog_both_expose(multi):
    r = router(*multi, expose="both")
    items = r.catalog({"openai": ["gpt-4o"]})
    ids = [i["id"] for i in items]
    assert ids == ["marg/openai/gpt-4o", "gpt-4o"]


def test_no_model_specified(multi):
    r = router(*multi)
    with pytest.raises(ApiError) as e:
        r.resolve(None)
    assert e.value.status == 400