"""`ApiError` and the end-of-stream sentinel."""

from __future__ import annotations

import pytest

from margAI import DONE, ApiError, DoneSentinel

# -- body ------------------------------------------------------------------


def test_the_default_body_is_openai_shaped():
    body = ApiError(429, "slow down", error_type="rate_limit_error", param="model", code="rate").body
    assert body == {
        "error": {
            "message": "slow down",
            "type": "rate_limit_error",
            "param": "model",
            "code": "rate",
        }
    }


def test_an_explicit_body_wins_over_the_fields():
    err = ApiError(500, "ignored", body={"error": {"message": "mine"}})
    assert err.body == {"error": {"message": "mine"}}


def test_with_body_copies_the_error_and_keeps_its_context():
    """The streaming path re-raises with this, so the clone has to carry the
    context too or error hooks and telemetry lose the call."""
    err = ApiError(502, "bad gateway")
    err.ctx = "the-call"
    clone = err.with_body({"error": {"message": "shaped"}})
    assert clone is not err
    assert clone.body == {"error": {"message": "shaped"}}
    assert clone.status == 502
    assert clone.ctx == "the-call"
    assert err.body != clone.body  # the original is untouched


# -- parsing an upstream payload ------------------------------------------


def test_a_normal_openai_error_body_parses():
    err = ApiError.from_openai_body({"error": {"message": "nope", "type": "invalid_request_error"}}, 400)
    assert (err.status, err.message, err.error_type) == (400, "nope", "invalid_request_error")


def test_an_anthropic_error_body_parses():
    """Anthropic nests once: {"type": "error", "error": {...}}."""
    body = {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
    err = ApiError.from_openai_body(body, 529)
    assert (err.status, err.message, err.error_type) == (529, "Overloaded", "overloaded_error")


def test_a_status_inside_the_body_wins_over_the_implicit_one():
    err = ApiError.from_openai_body({"error": {"message": "nope", "status": 402}}, 500)
    assert err.status == 402


@pytest.mark.parametrize("bad", ["not-a-number", "", None, [], {"nested": 1}, 1.5, True])
def test_a_non_numeric_status_falls_back_instead_of_raising(bad):
    """This runs on the failure path. An `int()` blowing up here would turn an
    upstream 4xx into an unhandled 500 and throw away the message it was
    trying to recover."""
    err = ApiError.from_openai_body({"error": {"message": "nope", "status": bad}}, 418)
    assert err.status == 418
    assert err.message == "nope"


def test_a_numeric_string_status_is_accepted():
    assert ApiError.from_openai_body({"error": {"message": "x", "status": " 429 "}}, 500).status == 429


@pytest.mark.parametrize("body", [None, "upstream exploded", 500, [1, 2], ""])
def test_a_non_dict_body_still_produces_a_usable_error(body):
    err = ApiError.from_openai_body(body, 502)
    assert err.status == 502
    assert isinstance(err.body["error"]["message"], str)


def test_a_bare_message_body_parses():
    err = ApiError.from_openai_body({"message": "quota exceeded"}, 429)
    assert (err.status, err.message) == (429, "quota exceeded")


# -- the end-of-stream sentinel -------------------------------------------


def test_the_sentinel_prints_the_same_way_everywhere():
    """`repr` said "<DONE>" and `str` said "DoneSentinel.TOKEN", so a log line
    and an f-string disagreed about the same value."""
    assert str(DONE) == repr(DONE) == "<DONE>"
    assert f"{DONE}" == "<DONE>"


def test_the_sentinel_is_exported_and_identity_comparable():
    from margAI.core import DONE as CORE_DONE
    from margAI.core import DoneSentinel as CoreDoneSentinel

    assert DONE is CORE_DONE
    assert DoneSentinel is CoreDoneSentinel
    assert DONE is DoneSentinel.TOKEN
