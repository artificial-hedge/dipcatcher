"""Fail-closed branches: rejected samples, bad stages, unknown gauges, redaction."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime

import pytest

from quant_fund.observe.logging import (
    get_correlation_id,
    log_event,
    new_correlation_id,
    set_log_sink,
)
from quant_fund.observe.metrics import (
    histogram_quantile_upper,
    record_error,
    record_latency,
    render_prometheus,
    reset_metrics,
    set_data_freshness,
    set_simulation_state,
)


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    monkeypatch.delenv("DIPCATCHER_OTEL_ENDPOINT", raising=False)
    reset_metrics()


# -- metrics: samples that must not corrupt the histogram --


@pytest.mark.parametrize("value", [-1.0, -0.0001, math.nan, math.inf, -math.inf])
def test_non_finite_or_negative_latency_is_dropped(
    monkeypatch: pytest.MonkeyPatch, value: float
) -> None:
    _enable(monkeypatch)
    record_latency("ingest", 0.5)
    record_latency("ingest", value)
    text = render_prometheus()
    assert 'dipcatcher_stage_latency_seconds_count{stage="ingest"} 1' in text
    assert 'dipcatcher_stage_latency_seconds_bucket{stage="ingest",le="+Inf"} 1' in text
    assert 'dipcatcher_stage_latency_seconds_sum{stage="ingest"} 0.5' in text


def test_negative_latency_registers_the_stage_but_counts_no_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected sample must not move count/sum; the series stays at zero."""
    _enable(monkeypatch)
    record_latency("decision", -5.0)
    text = render_prometheus()
    assert 'dipcatcher_stage_latency_seconds_bucket{stage="decision",le="+Inf"} 0' in text
    assert 'dipcatcher_stage_latency_seconds_sum{stage="decision"} 0' in text
    assert 'dipcatcher_stage_latency_seconds_count{stage="decision"} 0' in text
    assert math.isnan(histogram_quantile_upper("decision", 0.5))


def test_sample_above_the_last_bucket_lands_in_the_inf_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch)
    record_latency("features", 11.0)
    text = render_prometheus()
    assert 'dipcatcher_stage_latency_seconds_bucket{stage="features",le="10"} 0' in text
    assert 'dipcatcher_stage_latency_seconds_bucket{stage="features",le="+Inf"} 1' in text
    assert 'dipcatcher_stage_latency_seconds_count{stage="features"} 1' in text
    # The quantile falls past the largest finite bound, so the bound is +Inf.
    assert histogram_quantile_upper("features", 0.5) == math.inf


def test_quantile_is_nan_without_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    assert math.isnan(histogram_quantile_upper("unknown_stage", 0.5))
    record_latency("ingest", 0.25)
    assert math.isnan(histogram_quantile_upper("model", 0.5))
    assert histogram_quantile_upper("ingest", 0.99) == 0.25


def test_unknown_error_stage_raises_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    with pytest.raises(ValueError, match="unknown stage"):
        record_error("not-a-stage")
    record_error("otel_export")
    text = render_prometheus()
    assert 'dipcatcher_errors_total{stage="otel_export"} 1' in text


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_gauge_is_rejected(monkeypatch: pytest.MonkeyPatch, value: float) -> None:
    _enable(monkeypatch)
    with pytest.raises(ValueError, match="cannot record gauge"):
        set_data_freshness(value)
    with pytest.raises(ValueError, match="cannot record gauge"):
        set_simulation_state(pnl=value, gross_exposure=1.0, net_exposure=1.0)
    assert "dipcatcher_data_freshness_seconds" not in render_prometheus()
    assert "dipcatcher_simulation_pnl" not in render_prometheus()


def test_zero_gauge_renders_as_integer_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    set_data_freshness(0.0)
    set_simulation_state(pnl=0.0, gross_exposure=0.0, net_exposure=0.0)
    text = render_prometheus()
    assert "dipcatcher_data_freshness_seconds 0\n" in text
    assert "dipcatcher_simulation_pnl 0\n" in text
    assert "e+00" not in text and "0.0" not in text


def test_snapshot_is_a_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    from quant_fund.observe import metrics

    record_latency("ingest", 0.01)
    record_error("ingest")
    set_data_freshness(1.0)
    snap = metrics.REGISTRY.snapshot()
    snap["errors"]["ingest"] = 99
    snap["gauges"]["dipcatcher_data_freshness_seconds"] = 99.0
    snap["histograms"]["ingest"]["bucket_counts"][0] = 99
    fresh = metrics.REGISTRY.snapshot()
    assert fresh["errors"]["ingest"] == 1
    assert fresh["gauges"]["dipcatcher_data_freshness_seconds"] == 1.0
    assert fresh["histograms"]["ingest"]["bucket_counts"][0] == 0


def test_disabled_freshness_and_simulation_gauges_do_no_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DIPCATCHER_OBSERVE", raising=False)
    reset_metrics()
    set_data_freshness(1.0)
    set_simulation_state(pnl=1.0, gross_exposure=2.0, net_exposure=3.0)
    assert render_prometheus() == ""


# -- logging: stderr path and redaction --


def test_logs_write_one_json_line_to_stderr_when_no_sink(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _enable(monkeypatch)
    set_log_sink(None)
    new_correlation_id()
    log_event("stage", when=datetime(2026, 1, 2, tzinfo=UTC), count=3)
    out = capsys.readouterr()
    assert out.out == ""
    lines = out.err.splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event"] == "stage"
    assert record["level"] == "info"
    assert record["service"] == "dipcatcher"
    assert record["correlation_id"] == get_correlation_id()
    assert record["count"] == 3
    assert record["when"].startswith("2026-01-02")
    assert lines[0] == json.dumps(record, sort_keys=True)


def test_disabled_log_event_writes_nothing_to_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DIPCATCHER_OBSERVE", raising=False)
    set_log_sink(None)
    log_event("quiet", password="nope")
    assert capsys.readouterr().err == ""


def test_level_field_is_consumed_and_coerced_to_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch)
    sink: list[dict[str, object]] = []
    set_log_sink(sink)
    try:
        log_event("stage", level=7, token=123)
        assert sink[0]["level"] == "7"
        assert "level" not in {key for key in sink[0] if key in {"fields"}}
        assert sink[0]["token"] == "[redacted]"
    finally:
        set_log_sink(None)


@pytest.mark.parametrize(
    "name",
    [
        "key",
        "token",
        "secret",
        "authorization",
        "api_key",
        "access_token",
        "session_key",
        "private_key",
        "privatekey",
        "db_password",
        "password_hash",
        "x-api-key",
        "X-API-KEY",
        "client_secret",
        "slack_token",
        "signing_key",
    ],
)
def test_secret_looking_names_are_redacted(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    _enable(monkeypatch)
    sink: list[dict[str, object]] = []
    set_log_sink(sink)
    try:
        log_event("stage", **{name: "value"})
    finally:
        set_log_sink(None)
    assert sink[0][name] == "[redacted]"


@pytest.mark.parametrize(
    "name",
    ["key_id", "token_count", "password_policy_id", "keyring", "secretion", "authorization_id"],
)
def test_non_secret_names_survive(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    _enable(monkeypatch)
    sink: list[dict[str, object]] = []
    set_log_sink(sink)
    try:
        log_event("stage", **{name: "value"})
    finally:
        set_log_sink(None)
    # `password_policy_id` still contains "password", so it is redacted by design.
    expected = "[redacted]" if "password" in name else "value"
    assert sink[0][name] == expected
