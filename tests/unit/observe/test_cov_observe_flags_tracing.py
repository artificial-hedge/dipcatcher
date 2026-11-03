"""Flags, span guards and exporter guards that must never break the caller."""

from __future__ import annotations

import json

import pytest

from quant_fund.observe import flags
from quant_fund.observe.metrics import (
    record_latency,
    render_prometheus,
    reset_metrics,
    set_data_freshness,
    set_simulation_state,
)
from quant_fund.observe.serve import start_metrics_server, stop_metrics_server
from quant_fund.observe.tracing import (
    Span,
    _otlp,
    _otlp_attribute,
    clear_spans,
    finished_spans,
    span,
)


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    monkeypatch.delenv("DIPCATCHER_OTEL_ENDPOINT", raising=False)
    reset_metrics()
    clear_spans()


# -- flags --


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on", "On"])
def test_truthy_flag_spellings_enable_observability(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("DIPCATCHER_OBSERVE", value)
    assert flags.observe_enabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "enable", "2", "  "])
def test_other_flag_values_leave_it_off(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DIPCATCHER_OBSERVE", value)
    assert flags.observe_enabled() is False


@pytest.mark.parametrize("value", ["", "   ", "\n"])
def test_blank_endpoint_means_no_export(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DIPCATCHER_OTEL_ENDPOINT", value)
    assert flags.otel_endpoint() is None


def test_endpoint_and_host_are_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_OTEL_ENDPOINT", "  http://127.0.0.1:4318/v1/traces ")
    monkeypatch.setenv("DIPCATCHER_METRICS_HOST", " 0.0.0.0 ")
    assert flags.otel_endpoint() == "http://127.0.0.1:4318/v1/traces"
    assert flags.metrics_host() == "0.0.0.0"
    monkeypatch.setenv("DIPCATCHER_METRICS_HOST", "")
    assert flags.metrics_host() == "127.0.0.1"


@pytest.mark.parametrize("value", ["", "  "])
def test_missing_metrics_port_is_none(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DIPCATCHER_METRICS_PORT", value)
    assert flags.metrics_port() is None


@pytest.mark.parametrize("value,expected", [("0", 0), ("9464", 9464), ("65535", 65535)])
def test_in_range_metrics_port_values(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: int
) -> None:
    monkeypatch.setenv("DIPCATCHER_METRICS_PORT", value)
    assert flags.metrics_port() == expected


@pytest.mark.parametrize("value", ["-1", "65536", "100000"])
def test_out_of_range_metrics_port_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("DIPCATCHER_METRICS_PORT", value)
    with pytest.raises(ValueError, match="outside 0..65535"):
        flags.metrics_port()


def test_non_numeric_metrics_port_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_METRICS_PORT", "http")
    with pytest.raises(ValueError):
        flags.metrics_port()


# -- span guards --


def test_disabled_span_does_not_validate_the_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIPCATCHER_OBSERVE", raising=False)
    clear_spans()
    with span("not-a-stage") as current:
        assert current is None
    assert finished_spans() == []


def test_enabled_span_rejects_unknown_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    with pytest.raises(ValueError, match="unknown stage nope"), span("nope"):
        pass
    assert finished_spans() == []


def test_failed_span_records_an_error_and_still_records_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch)
    with pytest.raises(KeyError), span("features"):
        raise KeyError("boom")
    current = finished_spans()[-1]
    assert current.status == "error"
    assert current.end_time_unix_nano is not None
    text = render_prometheus()
    assert 'dipcatcher_errors_total{stage="features"} 1' in text
    assert 'dipcatcher_stage_latency_seconds_count{stage="features"} 1' in text


def test_finished_spans_are_capped_without_losing_the_newest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch)
    for _ in range(300):
        with span("ingest"):
            pass
    spans = finished_spans()
    assert len(spans) == 256
    assert spans[-1].name == "ingest"
    assert len({item.span_id for item in spans}) == 256


def test_non_primitive_attributes_are_stringified(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    with span("model", budget=None, label="x", count=3, ratio=0.5, flag=True):
        pass
    current = finished_spans()[-1]
    assert current.attributes == {
        "budget": "None",
        "label": "x",
        "count": 3,
        "ratio": 0.5,
        "flag": True,
    }


@pytest.mark.parametrize(
    "endpoint",
    ["ftp://127.0.0.1/v1/traces", "http:///v1/traces", "not-a-url", "127.0.0.1:4318"],
)
def test_unusable_endpoint_counts_one_export_error_and_never_raises(
    monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    _enable(monkeypatch)
    monkeypatch.setenv("DIPCATCHER_OTEL_ENDPOINT", endpoint)
    with span("decision") as current:
        assert current is not None
    assert 'dipcatcher_errors_total{stage="otel_export"} 1' in render_prometheus()


def test_otlp_payload_encodes_attribute_types() -> None:
    assert _otlp_attribute("b", True) == {"key": "b", "value": {"boolValue": True}}
    assert _otlp_attribute("i", 3) == {"key": "i", "value": {"intValue": "3"}}
    assert _otlp_attribute("f", 0.5) == {"key": "f", "value": {"doubleValue": 0.5}}
    assert _otlp_attribute("s", "x") == {"key": "s", "value": {"stringValue": "x"}}


def test_otlp_payload_omits_missing_end_and_parent() -> None:
    current = Span(
        name="ingest",
        trace_id="t" * 32,
        span_id="s" * 16,
        parent_span_id=None,
        start_time_unix_nano=5,
        correlation_id=None,
    )
    body = _otlp(current)["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert body["endTimeUnixNano"] == "5"
    assert body["status"]["code"] == 1
    assert "parentSpanId" not in body
    assert {"key": "correlation_id", "value": {"stringValue": ""}} in body["attributes"]
    assert json.dumps(body)  # JSON-serializable


# -- metrics server --


def test_stop_without_start_is_a_noop() -> None:
    stop_metrics_server()
    stop_metrics_server()


def test_second_start_returns_the_running_port(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    set_data_freshness(1.0)
    port = start_metrics_server(0)
    try:
        assert start_metrics_server(0) == port
        assert start_metrics_server(1234) == port
    finally:
        stop_metrics_server()


# -- gauge/registry interaction --


def test_gauges_render_help_and_type_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    set_data_freshness(2.0)
    set_simulation_state(pnl=-2.0, gross_exposure=4.0, net_exposure=-1.0)
    text = render_prometheus()
    assert text.count("# TYPE dipcatcher_data_freshness_seconds gauge") == 1
    assert text.count("# TYPE dipcatcher_simulation_pnl gauge") == 1
    assert "dipcatcher_simulation_pnl -2\n" in text
    assert "dipcatcher_simulation_net_exposure -1\n" in text


def test_latency_is_not_recorded_without_the_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIPCATCHER_OBSERVE", raising=False)
    reset_metrics()
    record_latency("not-a-stage", 1.0)
    assert render_prometheus() == ""
