"""Introspection: `Wrapper.describe()`, `/v1/margAI/tags`, and `margAI doctor`.

All three read the same payload on purpose. The failure this guards against is
a report built from the config rather than from the live registry: it would
happily describe a tag that was never registered, which is exactly the answer
an operator is asking the question to avoid getting.
"""

from __future__ import annotations

import asyncio
import io
import json

import pytest
from conftest import FakeTransport, make_wrapper, provider_config

from margAI.config import CostEntry, ModelPolicy
from margAI.doctor import ADVICE, ERROR, OK, WARN, main, run_doctor
from margAI.wrapper import Wrapper

PROVIDERS = (provider_config("openai", models=("gpt-4o",)),)


def busy() -> Wrapper:
    """A gateway with a pack, a virtual model and a cost table, so the report
    has something to report."""
    from margAI import install_bangtags

    app = make_wrapper(
        FakeTransport(),
        providers=PROVIDERS,
        models={"fast": ModelPolicy(name="fast", strategy="cheapest", max_cost_per_1m=1.0)},
        costs=(CostEntry("openai", "gpt-4o", 1.0, 3.0),),
    )
    install_bangtags(app, namespace="ep")

    with app.pack("demo", namespace="ep"):

        @app.on("audit")
        def audit(ctx, tag):
            """`audit` -- record the request for later review."""

    return app


# -- describe ---------------------------------------------------------------


def test_describe_reports_every_registered_tag():
    report = busy().describe()
    by_id = {entry["id"]: entry for entries in report["tags"].values() for entry in entries}
    assert "margAI:think" in by_id
    assert "ep:audit" in by_id
    assert by_id["ep:audit"]["packs"] == ["demo"]
    assert by_id["ep:audit"]["phases"] == ["before"]


def test_describe_carries_the_handler_docstring():
    """The docstring is the only documentation a pack author writes, and the
    first line is what a newcomer reads in the endpoint."""
    report = busy().describe()
    by_id = {entry["id"]: entry for entries in report["tags"].values() for entry in entries}
    assert by_id["ep:audit"]["doc"].startswith("`audit` -- record the request")


def test_describe_lists_virtual_models_and_providers():
    report = busy().describe()
    (model,) = report["virtual_models"]
    assert model["id"] == "margAI/fast"
    assert model["strategy"] == "cheapest"
    assert report["providers"][0]["name"] == "openai"
    assert report["providers"][0]["models"] == ["gpt-4o"]


def test_describe_reports_config_provenance():
    report = busy().describe()
    assert report["config"]["per_call_tables"] == ["packs", "models"]


def test_describe_names_the_undocumented_place_a_caller_cannot_reach():
    """Stating the boundary in the report means an operator learns it from the
    endpoint rather than from a rejected request."""
    assert busy().describe()["config"]["per_call_tables"] == ["packs", "models"]


def test_a_virtual_model_is_listed_in_the_models_catalog():
    import asyncio

    app = busy()
    catalog = asyncio.run(app.models())
    by_id = {item["id"]: item for item in catalog}
    assert "margAI/fast" in by_id
    assert by_id["margAI/fast"]["virtual"]["strategy"] == "cheapest"


# -- the HTTP endpoint -----------------------------------------------------


def test_the_tags_endpoint_serves_the_report():
    starlette = pytest.importorskip("starlette.testclient")
    from margAI.config import Config, GatewayConfig
    from margAI.transport.fastapi import build_app

    built = build_app(config=Config(gateway=GatewayConfig()))
    with starlette.TestClient(built) as client:
        response = client.get("/v1/margAI/tags")
    assert response.status_code == 200
    body = response.json()
    assert "margAI:think" in {e["id"] for e in body["tags"]["margAI"]}
    assert body["gateway"]["prefix"] == "margAI"


# -- the doctor -------------------------------------------------------------


def check(doctor, subject: str):
    return [c for c in doctor.checks if c.subject == subject]


EMPTY_PROVIDER = """
[providers.p]
kind = "openai"
base_url = "https://x.test/v1"
models = []
"""


ONE_PROVIDER = """
[providers.p]
kind = "openai"
base_url = "https://x.test/v1"
models = ["m"]
"""


def test_doctor_reports_ok_for_a_healthy_gateway():
    doctor = run_doctor(source=str(_write(ONE_PROVIDER)), env={})
    assert not doctor.failed
    assert any(c.level == OK for c in doctor.checks)


def test_doctor_fails_on_an_unreadable_config():
    doctor = run_doctor(source=str(_write('[gateway]\nport = "not-a-number"\n')), env={})
    assert doctor.failed
    assert check(doctor, "config")[0].level == ERROR


def test_doctor_names_the_config_file_and_the_layer_below_it():
    doctor = run_doctor(source=str(_write('[gateway]\nprefix = "x"\n')), env={})
    (entry,) = check(doctor, "config")
    assert entry.level == OK
    assert "embedded" in entry.message


def test_doctor_notes_a_missing_config_file(monkeypatch, tmp_path):
    """No ./margAI.toml in an empty directory is the normal no-config case."""
    monkeypatch.chdir(tmp_path)
    out = io.StringIO()
    run_doctor(env={}, out=out)
    assert "embedded defaults" in out.getvalue()


def test_a_named_config_that_is_absent_is_an_error_not_a_silent_default():
    """Falling through to the embedded defaults would start a gateway with
    none of the operator's providers and no indication why."""
    doctor = run_doctor(source=str(_nonexistent()), env={})
    assert doctor.failed
    assert "not found" in check(doctor, "config")[0].message

    doctor = run_doctor(source=None, env={"MARGAI_CONFIG": str(_nonexistent())})
    assert doctor.failed


def test_doctor_warns_about_a_provider_with_no_models():
    doctor = run_doctor(
        source=str(_write('[providers.empty]\nkind = "openai"\nbase_url = "https://x.test/v1"\nmodels = []\n')),
        env={},
    )
    assert any(c.level == WARN for c in check(doctor, "provider empty"))


def test_an_unusable_default_provider_is_an_error_not_a_warning():
    """Severity turns on what the provider is for: model-less and nothing
    defaults to it is dead weight, but the same provider behind
    gateway.default_provider is where every unqualified request lands."""
    doctor = run_doctor(
        source=str(_write('[gateway]\ndefault_provider = "p"\n' + EMPTY_PROVIDER)),
        env={},
    )
    (entry,) = [c for c in check(doctor, "provider p") if c.level == ERROR]
    assert "default_provider" in entry.message
    assert doctor.failed

    doctor = run_doctor(source=str(_write(EMPTY_PROVIDER)), env={})
    assert not [c for c in check(doctor, "provider p") if c.level == ERROR]
    assert [c for c in check(doctor, "provider p") if c.level == WARN]


def test_nothing_routable_is_an_error(monkeypatch):
    """The backstop behind every per-provider finding. Reaching it needs the
    embedded layer out of the way, because those providers always keep a
    `default_model` -- which is itself the reason an operator never sees this
    check fire on a default install."""
    from margAI import config as config_mod

    monkeypatch.setattr(config_mod, "default_config_raw", dict)
    doctor = run_doctor(
        source=str(_write('[gateway]\ndefault_provider = "p"\n' + EMPTY_PROVIDER)),
        env={},
    )
    assert any("no routable" in c.message for c in check(doctor, "routing"))


def test_the_embedded_layer_hides_an_emptied_provider():
    """Why a bare `models = []` is not enough to break a provider: the
    embedded table still supplies its default_model. Worth pinning, because
    the obvious fix -- "just clear the models list" -- silently does nothing.
    """
    doctor = run_doctor(
        source=str(
            _write(
                '[providers.openai]\nkind = "openai"\nbase_url = "https://x.test/v1"\nmodels = []\n'
                + ONE_PROVIDER
            )
        ),
        env={},
    )
    assert not [c for c in check(doctor, "provider openai") if c.level == ERROR]


def test_the_doctor_reports_on_the_gateway_that_would_actually_run():
    """`default_provider` reaches the router through `from_config`; a doctor
    that built its Wrapper by hand would silently never see it."""
    doctor = run_doctor(
        source=str(_write('[gateway]\ndefault_provider = "ghost"\n' + ONE_PROVIDER)),
        env={},
    )
    assert doctor.failed
    ghost = [c for c in check(doctor, "routing") if "ghost" in c.message]
    assert ghost
    assert ghost[0].level == ERROR


def test_doctor_warns_about_a_missing_api_key():
    doctor = run_doctor(
        source=str(
            _write(
                '[providers.p]\nkind = "openai"\nbase_url = "https://x.test/v1"\n'
                'models = ["m"]\napi_key_env = "DEFINITELY_NOT_SET_XYZ"\n'
            )
        ),
        env={},
    )
    (entry,) = check(doctor, "provider p")
    assert entry.level == WARN
    assert "DEFINITELY_NOT_SET_XYZ" in entry.message


def test_doctor_does_not_warn_about_a_keyless_local_provider():
    """Warning on every keyless local provider trains people to ignore the
    warning that matters."""
    doctor = run_doctor(
        source=str(
            _write(
                '[providers.local]\nkind = "openai-compatible"\n'
                'base_url = "http://127.0.0.1:8080/v1"\nmodels = ["m"]\n'
            )
        ),
        env={},
    )
    (entry,) = check(doctor, "provider local")
    assert entry.level == ADVICE


def test_doctor_errors_on_a_cost_ceiling_with_no_prices():
    """The failure is a 404 on every request to that model, which points at the
    model rather than at the config."""
    doctor = run_doctor(
        source=str(_write(ONE_PROVIDER + '\n[models.cheap]\nmax_cost_per_1m = 1.0\n')),
        env={},
    )
    (entry,) = check(doctor, "margAI/cheap")
    assert entry.level == ERROR
    assert "telemetry.costs" in entry.hint


def test_doctor_is_quiet_about_a_cost_ceiling_with_prices():
    doctor = run_doctor(
        source=str(
            _write(
                ONE_PROVIDER + '\n[models.cheap]\nmax_cost_per_1m = 1.0\n'
                '\n[telemetry.costs]\n"p/m" = { in = 0.1, out = 0.2 }\n'
            )
        ),
        env={},
    )
    assert not check(doctor, "margAI/cheap") or all(
        c.level != ERROR for c in check(doctor, "margAI/cheap")
    )


def test_doctor_errors_when_a_virtual_models_filters_exclude_everything():
    doctor = run_doctor(
        source=str(_write(ONE_PROVIDER + '\n[models.impossible]\nproviders = ["nobody"]\n')),
        env={},
    )
    (entry,) = check(doctor, "margAI/impossible")
    assert entry.level == ERROR


def test_doctor_notes_a_preference_that_matches_nothing():
    doctor = run_doctor(
        source=str(_write(ONE_PROVIDER + '\n[models.f]\nprefer = ["p/retired"]\n')),
        env={},
    )
    (entry,) = check(doctor, "margAI/f")
    assert entry.level == ADVICE
    assert "retired" in entry.message


def test_doctor_notes_undocumented_tags(monkeypatch):
    """An undocumented tag is discoverable by nothing, which is the whole
    point of registering one. A pack that registers a tag with no docstring is
    the case; the built-in tags all have one."""
    from margAI import wrapper as wrapper_mod
    from margAI.packs import PackRecord

    def install_silent(app, **kwargs):
        with app.pack("silent", namespace="quiet"):

            @app.on("mystery")
            def mystery(ctx, tag):
                pass

        return [PackRecord(name="silent", target="silent:install", installed=True, tags=("quiet:mystery",))]

    monkeypatch.setattr(wrapper_mod, "install_all", install_silent)
    doctor = run_doctor(source=str(_write(ONE_PROVIDER)), env={})
    undocumented = [c for c in doctor.checks if "no docstring" in c.message]
    assert [c.subject for c in undocumented] == ["quiet:mystery"]
    assert all(c.level == ADVICE for c in undocumented)


def test_documented_tags_raise_no_advice(monkeypatch):
    from margAI import wrapper as wrapper_mod
    from margAI.packs import PackRecord

    def install_terse(app, **kwargs):
        with app.pack("terse", namespace="quiet"):

            @app.on("known")
            def known(ctx, tag):
                """`known` -- it says what it does."""

        return [PackRecord(name="terse", target="terse:install", installed=True, tags=("quiet:known",))]

    monkeypatch.setattr(wrapper_mod, "install_all", install_terse)
    doctor = run_doctor(source=str(_write(ONE_PROVIDER)), env={})
    assert not [c for c in doctor.checks if "no docstring" in c.message]


def test_doctor_surfaces_a_broken_pack_by_name(monkeypatch):
    from margAI import wrapper as wrapper_mod
    from margAI.packs import PackFailure, PackRecord

    monkeypatch.setattr(
        wrapper_mod,
        "install_all",
        lambda app, **kw: [
            PackRecord(
                name="broken",
                target="broken:install",
                installed=False,
                failure=PackFailure("broken", "ImportError: no module named broken", "broken:install"),
            )
        ],
    )
    doctor = run_doctor(source=str(_write(ONE_PROVIDER)), env={})
    assert doctor.failed
    (entry,) = check(doctor, "pack broken")
    assert entry.level == ERROR
    assert "broken:install" in entry.hint


def test_doctor_warns_about_a_pack_that_registered_nothing(monkeypatch):
    from margAI import wrapper as wrapper_mod
    from margAI.packs import PackRecord

    monkeypatch.setattr(
        wrapper_mod,
        "install_all",
        lambda app, **kw: [PackRecord(name="quiet", target="quiet:install", installed=True)],
    )
    doctor = run_doctor(source=str(_write(ONE_PROVIDER)), env={})
    (entry,) = check(doctor, "pack quiet")
    assert entry.level == WARN


def test_doctor_json_output_is_machine_readable(capsys):
    main(["--json", "-c", str(_write(ONE_PROVIDER))])
    body = json.loads(capsys.readouterr().out)
    assert body["ok"] is True
    assert isinstance(body["checks"], list)


def test_doctor_never_sends_a_request():
    """A diagnostic with side effects is a diagnostic nobody trusts, and an
    offline CI run would be slow and flaky."""
    from margAI.doctor import _Offline

    offline = _Offline()
    with pytest.raises(RuntimeError, match="does not send requests"):
        asyncio.run(offline.request(None))
    with pytest.raises(RuntimeError, match="does not send requests"):
        asyncio.run(offline.open_stream(None))


def test_the_cli_dispatches_doctor():
    from margAI.__main__ import run

    assert run(["doctor", "--json", "-c", str(_write(ONE_PROVIDER))]) in (0, 1)


def test_the_cli_rejects_an_unknown_subcommand(capsys):
    from margAI.__main__ import run

    assert run(["frobnicate"]) == 2
    assert "usage" in capsys.readouterr().err


def test_the_cli_help_lists_doctor(capsys):
    from margAI.__main__ import run

    assert run(["--help"]) == 0
    assert "doctor" in capsys.readouterr().out


def _write(text: str):
    """Write a config file and return its path.

    Takes TOML text rather than a dict so each test's config reads as the
    config an operator would actually write -- and so a malformed fixture fails
    here, at the line that wrote it, instead of surfacing as a doctor finding
    about a file the test never meant to create.
    """
    import pathlib
    import tempfile
    import tomllib

    path = pathlib.Path(tempfile.mkdtemp()) / "margAI.toml"
    path.write_text(text)
    tomllib.loads(text)
    return path


def _nonexistent():
    import pathlib
    import tempfile

    return pathlib.Path(tempfile.mkdtemp()) / "absent.toml"
