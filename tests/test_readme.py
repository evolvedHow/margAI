"""The README's examples and the `examples/` scripts, executed.

Documentation that is never run is documentation that is wrong, and a green test
suite is exactly the thing that lets it stay wrong. Every item below was a real
defect found by running what the docs say:

- `rag=quantum computing` in the Quick Start. A directive splits on commas and
  then on whitespace, so the space made `computing` a second tag, the handler
  never saw the query, and the example quietly did nothing.
- `ctx.route_to("local/llama3.2-1b")` in two examples. The catalog exposes
  `<namespace>/<provider>/<model>`, so the short form is a 404.
- `examples/margAI.toml` declaring `api_key_env = ""`, which the loader
  rejects -- the example could not start.
- `examples/rag_gateway.py` pointing at `examples/simple_gateway.toml`, a path
  that only resolves from the repo root.
- `@app.on_request` reported missing from three examples that were using it
  correctly, because the check that reported it looked only at `Wrapper` and
  `on_request` is a `Gateway` method.

The last one is the reason the checks are behavioural. `hasattr` on the wrong
class proves the opposite of the truth.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import re
import sys

import pytest
from conftest import FakeTransport, UpstreamResponse, chat_payload, make_wrapper, provider_config

from margAI import install_bangtags
from margAI.bangtag import find_directive

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"


# -- a tag value cannot contain a space ---------------------------------------


def test_a_directive_splits_on_commas_then_whitespace():
    """The grammar, stated as a test, because every example depends on it."""
    _, tags = find_directive("!app: rag=quantum-computing, summarize", "app")
    assert [(t.name, t.value) for t in tags] == [
        ("rag", "quantum-computing"),
        ("summarize", None),
    ]


def test_a_space_in_a_value_splits_into_another_tag():
    """Not a limitation to work around silently -- the thing the README got
    wrong. Worth pinning so the fix is a decision, not an accident."""
    _, tags = find_directive("!app: rag=quantum computing, summarize", "app")
    assert [(t.name, t.value) for t in tags] == [
        ("rag", "quantum"),
        ("computing", None),
        ("summarize", None),
    ]


def test_quoting_does_not_protect_a_spaces_value():
    """Someone's first guess at the fix above. It does not work, and it
    half-works in the worst way: the value keeps the opening quote and the rest
    becomes a junk tag."""
    _, tags = find_directive('!app: rag="quantum computing"', "app")
    assert [(t.name, t.value) for t in tags] == [
        ("rag", '"quantum'),
        ('computing"', None),
    ]


def test_no_readme_example_contains_a_spaced_tag_value():
    """The check that would have caught the original bug."""

    offenders = [
        line
        for line in README.read_text().splitlines()
        if re.match(r"^\s*![A-Za-z0-9_-]+:", line) and re.search(r"[\w-]+=[^,]*\s", line)
    ]
    assert offenders == []


# -- the Quick Start example, end to end --------------------------------------


def test_the_quick_start_example_runs():
    """`Gateway`'s three README tags, fired exactly as the README fires them."""
    from margAI.config import GatewayConfig

    transport = FakeTransport([UpstreamResponse(200, chat_payload("ANSWER"))])
    app = make_wrapper(transport, gateway=GatewayConfig(prefix="app"))
    install_bangtags(app, namespace="app")

    @app.on("summarize", namespace="app")
    def summarize(ctx, tag):
        ctx.add_system("Provide a brief summary.")

    @app.on("rag", namespace="app")
    def add_context(ctx, tag):
        ctx.add_system(f"Relevant context:\ndocs-for-{tag.value}")

    response = asyncio.run(
        app.complete(
            {
                "model": "app/openai/gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": "!app: rag=quantum-computing, summarize\n\nWhat is quantum entanglement?",
                    }
                ],
            }
        )
    )

    assert response.status == 200

    sent = transport.requested[0].json
    system = "\n".join(m["content"] for m in sent["messages"] if m["role"] == "system")
    assert "Provide a brief summary." in system
    assert "docs-for-quantum-computing" in system

    # The directive is consumed, not forwarded: the provider must never see the
    # `!app:` line, or a model that takes instructions literally will start
    # answering to a tag.
    user_text = "".join(m["content"] for m in sent["messages"] if m["role"] == "user")
    assert "!app:" not in user_text
    assert "quantum entanglement" in user_text


# -- the model ids the README tells people to route to ------------------------


def test_every_model_id_in_the_readme_is_namespaced():
    """`ctx.route_to("local/llama")` 404s. The catalog exposes
    `<namespace>/<provider>/<model>`, and a README that teaches the short form
    teaches a 404."""
    for model in re.findall(r'route_to\("([^"]+)"\)', README.read_text()):
        assert model.count("/") >= 2, f"README routes to un-namespaced {model!r}"


def test_a_namespaced_id_in_the_readme_resolves_against_a_real_catalog():
    """Not just well-shaped: present in `/v1/models`, which is what
    `ctx.route_to` resolves against. Configured with the two providers the
    README's examples actually name, so this fails if an id drifts."""
    app = make_wrapper(
        FakeTransport(),
        providers=(
            provider_config("openai", models=("gpt-4o",)),
            provider_config("local", base_url="http://llama.test/v1", models=("llama3.2-1b",)),
        ),
    )
    ids = {m["id"] for m in asyncio.run(app.models())}
    for model in re.findall(r'route_to\("([^"]+)"\)', README.read_text()):
        assert model in ids, f"{model!r} is not in the model catalog"


# -- decorators the README names ----------------------------------------------


def test_every_decorator_the_readme_names_exists():
    """A name that *looks* right is the failure mode a docstring cannot catch.

    Note both front ends are checked for every name. `@app.on_request` is a
    `Gateway` method and never was a `Wrapper` one, so a check that only looks
    at `Wrapper` "proves" the opposite of the truth -- which is how
    `@app.on_request` got reported missing while three examples were using it
    correctly.
    """
    from margAI import Gateway, Wrapper

    for name in ("before", "after", "stream", "error", "on", "marglet", "add_marglet"):
        assert hasattr(Wrapper, name), f"Wrapper.{name}"
        assert hasattr(Gateway, name), f"Gateway.{name}"

    for name in ("tag", "serve", "on_request", "on_response", "on_stream", "on_error"):
        assert hasattr(Gateway, name), f"Gateway.{name}"


def _gateway():
    from margAI import Gateway
    from margAI.config import Config, GatewayConfig

    return Gateway(Config(gateway=GatewayConfig()))


@pytest.mark.parametrize(
    ("outer", "inner"),
    [("on_request", "before"), ("on_response", "after"), ("on_stream", "stream"), ("on_error", "error")],
)
def test_the_gateway_hook_aliases_reach_the_same_hook(monkeypatch, outer: str, inner: str):
    """`on_request` and `before` are two spellings of one registration. If they
    ever diverged, an example using one and a README using the other would be
    testing different pipelines."""
    calls: list[tuple] = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return lambda fn: fn

    def handler(*args, **kwargs):
        return None

    app = _gateway()
    monkeypatch.setattr(app.wrapper, inner, record)
    getattr(app, outer)(handler)
    assert len(calls) == 1, outer

    # And the same registration is the one the `Wrapper` spelling makes.
    other = _gateway()
    monkeypatch.setattr(other.wrapper, inner, record)
    getattr(other, inner)(handler)
    assert len(calls) == 2, inner
    assert calls[0] == calls[1], f"{outer} and {inner} registered differently"


@pytest.mark.parametrize(
    "name",
    ["before", "after", "stream", "error", "on", "marglet", "add_marglet", "pack"],
)
def test_a_proxy_on_gateway_forwards_to_the_wrapper(monkeypatch, name: str):
    """The proxies are written out on purpose, so a typo is an `AttributeError`
    at the call site instead of a `__getattr__` that guesses."""
    seen: list[tuple] = []
    app = _gateway()
    # Recording the call *is* the evidence of delegation; the real function is
    # not called because its own arguments are its business.
    monkeypatch.setattr(app.wrapper, name, lambda *a, **k: seen.append((a, k)))
    getattr(app, name)("x", flag=1)
    assert seen == [(("x",), {"flag": 1})], name


def test_gateway_does_not_invent_methods():
    app = _gateway()
    with pytest.raises(AttributeError):
        _ = app.definitely_not_a_method


# -- the examples -------------------------------------------------------------


EXAMPLES = pathlib.Path(__file__).resolve().parent.parent / "examples"


def _run_example(name: str) -> None:
    """Import an example the way a reader would: by path, from the repo root.

    Not `import examples.x` -- that would resolve differently than the command in
    the README, and the point is to check the command that is printed.
    """
    import importlib.util

    for env in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.setdefault(env, "sk-test")

    spec = importlib.util.spec_from_file_location(f"_example_{name}", EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize("name", sorted(p.stem for p in EXAMPLES.glob("*.py")))
def test_an_example_actually_starts(name: str):
    """Three of these were broken and none of it showed up in the test suite.

    `@app.on_request` and `@app.on_response` were documented in three examples
    and have never existed on `Wrapper` or `Gateway`; one example's config
    declared `api_key_env = ""`, which the loader rejects; another pointed at
    `examples/simple_gateway.toml`, a path that only resolves from the repo
    root. Every one of them is a `file.py, line N` failure for the first person
    who tries the example, and a green test suite is exactly the thing that
    makes it worse.
    """
    _run_example(name)


# -- licensing ---------------------------------------------------------------


ROOT = pathlib.Path(__file__).resolve().parent.parent


def _pyproject() -> str:
    return (ROOT / "pyproject.toml").read_text()


def test_the_declared_license_is_the_one_in_the_license_file():
    """Three places name the license: the `license` key, the trove classifier,
    and the text of `LICENSE`. They drifted once already during release prep --
    the classifier said MIT while the file said Apache-2.0 would have been a
    guess about to be made. Nothing about the build fails when they disagree;
    PyPI just renders the wrong thing."""
    import re

    pyproject = _pyproject()
    declared = re.search(r'^license = "(.+?)"', pyproject, re.M).group(1)
    classifier = re.search(r'"License :: OSI Approved :: (.+?)"', pyproject).group(1)

    assert declared == "Apache-2.0"
    assert classifier == "Apache Software License"
    assert "Apache License" in (ROOT / "LICENSE").read_text()


def test_a_notice_file_exists_because_section_4d_needs_one():
    """Apache-2.0 §4(d) makes the NOTICE contents an obligation on anyone
    redistributing. Shipping a license with no NOTICE makes that obligation
    refer to nothing, which quietly makes the attribution promise hollow."""
    notice = ROOT / "NOTICE"
    assert notice.exists()
    text = notice.read_text()
    assert "Vish Ganapathy" in text
    # It has to be in the distributions, not just the repository.
    assert '"NOTICE"' in _pyproject()
    assert 'license-files = ["LICENSE", "NOTICE"]' in _pyproject()


def test_the_author_is_in_the_metadata():
    """An `authors` entry is what PyPI renders as the package author, and it is
    the only place a visitor sees who to hold responsible."""
    import re

    match = re.search(r"authors = \[\s*\{ name = \"(.+?)\", email = \"(.+?)\"", _pyproject())
    assert match, "no authors entry"
    assert match.group(1) == "Vish Ganapathy"
    assert match.group(2) == "vish.ganapathy@gmail.com"


def test_the_community_files_are_present_and_linked():
    """A public repo that ships a CoC nobody can find, or a CONTRIBUTING that
    documents the wrong commands, is worse than not having them."""
    for name in ("CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "SECURITY.md", "RELEASING.md"):
        assert (ROOT / name).exists(), name

    readme = README.read_text()
    assert "CONTRIBUTING.md" in readme
    assert "CODE_OF_CONDUCT.md" in readme
    assert "SECURITY.md" in readme
    assert "Apache-2.0" in readme

    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    # The commands CONTRIBUTING tells people to run have to be the ones CI runs.
    assert "uv run pytest -q" in contributing
    assert "uv run ruff check src tests" in contributing
    assert "uv run mypy src/margAI" in contributing


def test_the_version_lives_in_exactly_one_place():
    """`pyproject.toml` reads the version from `__init__.py`. Two copies is one
    copy to forget at tag time, and the failure is a release published under a
    version nobody chose."""
    import re

    pyproject = _pyproject()
    assert 'dynamic = ["version"]' in pyproject
    assert not re.search(r"^version = ", pyproject, re.M), "version is hardcoded again"
    assert 'path = "src/margAI/__init__.py"' in pyproject

    from margAI import __version__

    assert re.match(r"^\d+\.\d+\.\d+", __version__)
    assert f'__version__ = "{__version__}"' in (ROOT / "src" / "margAI" / "__init__.py").read_text()


def test_the_publish_workflow_is_separate_from_ci():
    """PyPI trusts a *workflow file*. Trusting `ci.yml` would mean anyone who
    can edit a lint step can also edit the publish path, so the release is its
    own file -- and PyPI's guidance is to name it `release.yml`.

    Also guards the bug that is easy to reintroduce: a tag-triggered workflow
    cannot see `ci.yml`'s artifacts, because `download-artifact` only reads the
    current run. The build has to live in the release workflow itself, or the
    first release fails.
    """
    import yaml

    workflows = ROOT / ".github" / "workflows"
    release_path = workflows / "release.yml"
    assert release_path.exists(), "release.yml is missing"

    ci = (workflows / "ci.yml").read_text()
    release = yaml.safe_load(release_path.read_text())

    assert "pypi-publish" not in ci, "publishing crept back into ci.yml"

    assert sorted(release["jobs"]) == ["build", "publish"], "publish must run off a build in this file"
    assert release["jobs"]["publish"]["needs"] == "build"

    # The signed job is the least-privileged one: two steps, no checkout, no build.
    publish = release["jobs"]["publish"]
    assert publish["permissions"] == {"id-token": "write"}
    assert publish["environment"]["name"] == "pypi"
    steps = publish["steps"]
    assert len(steps) == 2, f"publish job should be two steps, has {len(steps)}"
    assert steps[0]["uses"].startswith("actions/download-artifact")
    assert steps[1]["uses"].startswith("pypa/gh-action-pypi-publish")
    # Checkout or build here would mean the credential-holding job runs code.
    assert not any("checkout" in (s.get("uses") or "") for s in steps)

    # The tag and the packaged version must agree, or a version gets burned on
    # the wrong name.
    build_steps = release["jobs"]["build"]["steps"]
    assert any("GITHUB_REF_NAME" in (s.get("run") or "") for s in build_steps), "no tag/version check"


def test_the_type_check_scope_agrees_across_readme_contributing_and_ci():
    """`uv run mypy src tests` was documented in the README and fails on a clean
    tree with 658 `no-untyped-def` errors, because the suite is deliberately not
    annotated end to end. One command, three places, and they had drifted."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    scope = "uv run mypy src/margAI"

    assert scope in ci
    assert scope in (ROOT / "CONTRIBUTING.md").read_text()
    assert scope in README.read_text()

    # And the wrong one is gone from both documents.
    assert "uv run mypy src tests" not in README.read_text()
    assert "uv run mypy src tests" not in (ROOT / "CONTRIBUTING.md").read_text()
