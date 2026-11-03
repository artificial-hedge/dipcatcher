"""Bounded operational counters for live execution telemetry."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import pytest

from quant_fund.observe.metrics import (
    REGISTRY,
    record_live_fill_record,
    record_live_kill_switch_transition,
    record_live_order_recovery,
    record_live_order_result,
    record_live_order_submit,
    record_live_pretrade_denial,
    render_prometheus,
    reset_metrics,
    set_data_freshness,
)
from quant_fund.pretrade.codes import REASON_ORDER

_VENUES = ("loopback", "drop", "shadow", "alpaca-paper", "alpaca-live")
_STATUSES = (
    "accepted",
    "filled",
    "partially_filled",
    "canceled",
    "rejected",
    "not_acknowledged",
    "expired",
    "replaced",
    "done_for_day",
    "pending_cancel",
    "pending_replace",
    "suspended",
    "unmapped",
    "response_lost",
)
_VERDICTS = ("accepted", "out_of_order_fill", "replaced_provisional", "orphan_fill")
_KINDS = ("order", "cancel")
_REASONS = tuple(name for _, name in REASON_ORDER) + ("other",)
_STATES = ("latched", "reset")
_OUTCOMES = ("recovered", "not_found", "error")

_RECORDERS = (
    record_live_order_submit,
    record_live_order_result,
    record_live_fill_record,
    record_live_pretrade_denial,
    record_live_kill_switch_transition,
    record_live_order_recovery,
)
_INVALID_CALLS: tuple[tuple[Callable[..., None], dict[str, str]], ...] = (
    (record_live_order_submit, {"venue": "invalid-venue"}),
    (record_live_order_result, {"venue": "loopback", "status": "invalid-status"}),
    (record_live_fill_record, {"venue": "loopback", "verdict": "invalid-verdict"}),
    (
        record_live_pretrade_denial,
        {"venue": "loopback", "kind": "invalid-kind", "primary_reason": "other"},
    ),
    (
        record_live_pretrade_denial,
        {"venue": "loopback", "kind": "order", "primary_reason": "invalid-reason"},
    ),
    (
        record_live_kill_switch_transition,
        {"venue": "loopback", "state": "invalid-state"},
    ),
    (record_live_order_recovery, {"venue": "loopback", "outcome": "invalid-outcome"}),
)


def _enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIPCATCHER_OBSERVE", "1")
    reset_metrics()


def test_live_counter_apis_are_keyword_only() -> None:
    for recorder in _RECORDERS:
        parameters = inspect.signature(recorder).parameters.values()
        assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters)


def test_live_counters_accumulate_in_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)
    record_live_order_submit(venue="loopback")
    record_live_order_submit(venue="loopback")
    record_live_order_result(venue="loopback", status="accepted")
    record_live_order_result(venue="loopback", status="accepted")
    record_live_order_result(venue="alpaca-paper", status="response_lost")
    record_live_fill_record(venue="shadow", verdict="orphan_fill")
    record_live_pretrade_denial(
        venue="alpaca-live",
        kind="cancel",
        primary_reason="kill_switch",
    )
    record_live_kill_switch_transition(venue="alpaca-live", state="latched")
    record_live_order_recovery(venue="drop", outcome="not_found")

    assert REGISTRY.snapshot()["counters"] == {
        "dipcatcher_live_order_submits_total": {("loopback",): 2},
        "dipcatcher_live_order_results_total": {
            ("loopback", "accepted"): 2,
            ("alpaca-paper", "response_lost"): 1,
        },
        "dipcatcher_live_fill_records_total": {("shadow", "orphan_fill"): 1},
        "dipcatcher_live_pretrade_denials_total": {
            ("alpaca-live", "cancel", "kill_switch"): 1,
        },
        "dipcatcher_live_kill_switch_transitions_total": {
            ("alpaca-live", "latched"): 1,
        },
        "dipcatcher_live_order_recovery_attempts_total": {("drop", "not_found"): 1},
    }


def test_prometheus_has_exact_labels_and_appends_counters_after_gauges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch)
    set_data_freshness(2.5)
    record_live_order_recovery(venue="drop", outcome="not_found")
    record_live_kill_switch_transition(venue="alpaca-live", state="latched")
    record_live_pretrade_denial(
        venue="alpaca-live",
        kind="cancel",
        primary_reason="stale_data",
    )
    record_live_fill_record(venue="shadow", verdict="orphan_fill")
    record_live_order_result(venue="loopback", status="partially_filled")
    record_live_order_submit(venue="loopback")
    record_live_order_submit(venue="alpaca-live")

    lines = render_prometheus().splitlines()
    samples = [line for line in lines if not line.startswith("#")]
    assert samples == [
        "dipcatcher_data_freshness_seconds 2.5",
        'dipcatcher_live_order_submits_total{venue="alpaca-live"} 1',
        'dipcatcher_live_order_submits_total{venue="loopback"} 1',
        'dipcatcher_live_order_results_total{venue="loopback",status="partially_filled"} 1',
        'dipcatcher_live_fill_records_total{venue="shadow",verdict="orphan_fill"} 1',
        (
            'dipcatcher_live_pretrade_denials_total{venue="alpaca-live",kind="cancel",'
            'primary_reason="stale_data"} 1'
        ),
        ('dipcatcher_live_kill_switch_transitions_total{venue="alpaca-live",state="latched"} 1'),
        'dipcatcher_live_order_recovery_attempts_total{venue="drop",outcome="not_found"} 1',
    ]

    metric_names = (
        "dipcatcher_live_order_submits_total",
        "dipcatcher_live_order_results_total",
        "dipcatcher_live_fill_records_total",
        "dipcatcher_live_pretrade_denials_total",
        "dipcatcher_live_kill_switch_transitions_total",
        "dipcatcher_live_order_recovery_attempts_total",
    )
    for name in metric_names:
        help_index = next(i for i, line in enumerate(lines) if line.startswith(f"# HELP {name} "))
        type_index = lines.index(f"# TYPE {name} counter")
        sample_index = next(i for i, line in enumerate(lines) if line.startswith(f"{name}{{"))
        assert help_index < type_index < sample_index

    help_lines = [line.lower() for line in lines if line.startswith("# HELP dipcatcher_live_")]
    assert len(help_lines) == len(metric_names)
    assert all("operational telemetry" in line for line in help_lines)
    assert all("not a live p&l claim" in line for line in help_lines)
    assert any("not unique economic executions" in line for line in help_lines)
    assert any("one primary reason per decision" in line for line in help_lines)


def test_reset_clears_live_counters_and_restores_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch)
    record_live_order_submit(venue="loopback")
    assert render_prometheus()

    reset_metrics()

    assert REGISTRY.snapshot() == {
        "histograms": {},
        "errors": {},
        "gauges": {},
        "counters": {},
    }
    assert render_prometheus() == ""


def test_disabled_live_counters_ignore_invalid_values_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DIPCATCHER_OBSERVE", raising=False)
    reset_metrics()

    for recorder, kwargs in _INVALID_CALLS:
        assert recorder(**kwargs) is None

    assert REGISTRY.snapshot()["counters"] == {}
    assert render_prometheus() == ""


@pytest.mark.parametrize(("recorder", "kwargs"), _INVALID_CALLS)
def test_enabled_live_counters_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    recorder: Callable[..., None],
    kwargs: dict[str, str],
) -> None:
    _enable(monkeypatch)

    with pytest.raises(ValueError):
        recorder(**kwargs)

    assert REGISTRY.snapshot()["counters"] == {}
    assert render_prometheus() == ""


def test_live_counter_output_is_bounded_and_secret_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable(monkeypatch)
    for venue in _VENUES:
        record_live_order_submit(venue=venue)
    for venue, status in product(_VENUES, _STATUSES):
        record_live_order_result(venue=venue, status=status)
    for venue, verdict in product(_VENUES, _VERDICTS):
        record_live_fill_record(venue=venue, verdict=verdict)
    for venue, kind, reason in product(_VENUES, _KINDS, _REASONS):
        record_live_pretrade_denial(venue=venue, kind=kind, primary_reason=reason)
    for venue, state in product(_VENUES, _STATES):
        record_live_kill_switch_transition(venue=venue, state=state)
    for venue, outcome in product(_VENUES, _OUTCOMES):
        record_live_order_recovery(venue=venue, outcome=outcome)

    secret = "live-metrics-test-secret"
    with pytest.raises(ValueError):
        record_live_order_submit(venue=secret)
    with pytest.raises(ValueError):
        record_live_pretrade_denial(venue="loopback", kind="order", primary_reason=secret)

    snapshot = REGISTRY.snapshot()
    counters = snapshot["counters"]
    assert sum(len(series) for series in counters.values()) == 400
    text = render_prometheus()
    assert len([line for line in text.splitlines() if not line.startswith("#")]) == 400
    assert len(text.splitlines()) == 412
    assert secret not in text
    assert secret not in repr(snapshot)


def test_live_counter_updates_are_thread_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable(monkeypatch)

    def record_batch(_: int) -> None:
        for _ in range(50):
            record_live_order_submit(venue="loopback")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(record_batch, index) for index in range(8)]
        for future in futures:
            future.result()

    counters = REGISTRY.snapshot()["counters"]
    assert counters["dipcatcher_live_order_submits_total"][("loopback",)] == 400
