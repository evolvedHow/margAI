"""Telemetry sinks: the file emitter and its generated log name."""

from __future__ import annotations

import json
import re

from margAI.config import TelemetryConfig
from margAI.telemetry import Telemetry


def test_file_sink_writes_one_json_line_per_record(tmp_path):
    tele = Telemetry(
        TelemetryConfig(enabled=True, emit="file", dir=str(tmp_path), label="vedanta")
    )
    tele.emit(tele.record(source="chat", provider="local", upstream_model="m", status=200))
    tele.emit(
        tele.record(source="chat", provider="local", upstream_model="m", status=500, error="boom")
    )
    tele.close()

    assert tele.path is not None
    assert re.fullmatch(r"vedanta_telemetry_\d{12}\.log", tele.path.name)
    lines = tele.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["status"] == 200
    assert json.loads(lines[1])["error"] == "boom"


def test_caller_label_names_the_file_when_telemetry_label_is_unset(tmp_path):
    # The gateway passes its prefix as the label; with no [telemetry] label,
    # that is what names the log, so each app is distinguishable in one dir.
    tele = Telemetry(TelemetryConfig(emit="file", dir=str(tmp_path)), label="veda")
    assert tele.path is not None
    assert tele.path.name.startswith("veda_telemetry_")
    tele.close()


def test_telemetry_label_wins_over_caller_and_is_sanitised(tmp_path):
    tele = Telemetry(
        TelemetryConfig(emit="file", dir=str(tmp_path), label="My App!"), label="prefix"
    )
    assert tele.path is not None
    assert tele.path.name.startswith("My-App_telemetry_")
    tele.close()


def test_file_sink_is_appendable_and_disabled_emit_writes_nothing(tmp_path):
    off = Telemetry(TelemetryConfig(enabled=False, emit="file", dir=str(tmp_path), label="x"))
    off.emit(off.record(status=200))
    off.close()
    assert list(tmp_path.iterdir()) == []

    tele = Telemetry(TelemetryConfig(emit="file", dir=str(tmp_path), label="x"))
    tele.emit(tele.record(status=200))
    tele.close()
    path = tele.path
    assert path is not None
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1
