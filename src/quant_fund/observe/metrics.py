"""In-process Prometheus metrics. No client library and no work when disabled.

Gauges that describe simulation P&L or exposure are labeled as simulation in
their help text. They are operational telemetry, not research headlines, and
they are not a live P&L claim.
"""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable
from typing import Any

from quant_fund.observe.flags import STAGES, observe_enabled

BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_ERROR_STAGES = frozenset((*STAGES, "otel_export"))

METRIC_HELP = {
    "dipcatcher_stage_latency_seconds": "Latency of an instrumented stage, in seconds.",
    "dipcatcher_errors_total": "Errors observed in an instrumented stage.",
    "dipcatcher_data_freshness_seconds": "Age of the newest input used by a stage, in seconds.",
    "dipcatcher_simulation_pnl": (
        "Simulated mark-to-market result. Not a live P&L claim and not a research score."
    ),
    "dipcatcher_simulation_gross_exposure": (
        "Simulated gross exposure. Not a live trading position and not a research score."
    ),
    "dipcatcher_simulation_net_exposure": (
        "Simulated net exposure. Not a live trading position and not a research score."
    ),
    "dipcatcher_ready": "1 when the lab process may accept work. 0 when readiness is closed.",
    "dipcatcher_draining": "1 when the lab SIGTERM drain latch is set.",
    "dipcatcher_kill_switch_blocking": "1 when the operator kill file blocks new orders.",
    "dipcatcher_kill_switch_info": (
        "Current operator kill code. The code label is a stable reader code, never the file body."
    ),
}

_OPS_CODE = re.compile(r"^[A-Za-z0-9_:-]{1,64}$")
_OpsSampler = Callable[[], tuple[float, float, float, str]]
_OPS_SAMPLER: _OpsSampler | None = None

_COUNTER_HELP = {
    "dipcatcher_live_order_submits_total": (
        "Operational telemetry for live order submissions; not a live P&L claim."
    ),
    "dipcatcher_live_order_results_total": (
        "Operational telemetry for live order results; not a live P&L claim."
    ),
    "dipcatcher_live_fill_records_total": (
        "Operational telemetry for fill records, which are not unique economic executions; "
        "not a live P&L claim."
    ),
    "dipcatcher_live_pretrade_denials_total": (
        "Operational telemetry for pre-trade denials, with one primary reason per decision; "
        "not a live P&L claim."
    ),
    "dipcatcher_live_kill_switch_transitions_total": (
        "Operational telemetry for live kill-switch transitions; not a live P&L claim."
    ),
    "dipcatcher_live_order_recovery_attempts_total": (
        "Operational telemetry for live order recovery attempts; not a live P&L claim."
    ),
}

_COUNTER_LABELS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "dipcatcher_live_order_submits_total": (
        ("venue", ("loopback", "drop", "shadow", "alpaca-paper", "alpaca-live")),
    ),
    "dipcatcher_live_order_results_total": (
        ("venue", ("loopback", "drop", "shadow", "alpaca-paper", "alpaca-live")),
        (
            "status",
            (
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
            ),
        ),
    ),
    "dipcatcher_live_fill_records_total": (
        ("venue", ("loopback", "drop", "shadow", "alpaca-paper", "alpaca-live")),
        (
            "verdict",
            ("accepted", "out_of_order_fill", "replaced_provisional", "orphan_fill"),
        ),
    ),
    "dipcatcher_live_pretrade_denials_total": (
        ("venue", ("loopback", "drop", "shadow", "alpaca-paper", "alpaca-live")),
        ("kind", ("order", "cancel")),
        (
            "primary_reason",
            (
                "kill_switch",
                "stale_data",
                "non_finite",
                "market_closed",
                "halted",
                "max_order_quantity",
                "max_order_notional",
                "price_collar",
                "position_quantity",
                "position_notional",
                "gross_exposure",
                "net_exposure",
                "concentration",
                "duplicate_order",
                "order_rate",
                "message_rate",
                "reg_sho",
                "pattern_day_trader",
                "good_faith_violation",
                "daily_loss",
                "trailing_drawdown",
                "internal_error",
                "buying_power",
                "state_full",
                "unknown_symbol",
                "non_monotonic_time",
                "manual_kill",
                "other",
            ),
        ),
    ),
    "dipcatcher_live_kill_switch_transitions_total": (
        ("venue", ("loopback", "drop", "shadow", "alpaca-paper", "alpaca-live")),
        ("state", ("latched", "reset")),
    ),
    "dipcatcher_live_order_recovery_attempts_total": (
        ("venue", ("loopback", "drop", "shadow", "alpaca-paper", "alpaca-live")),
        ("outcome", ("recovered", "not_found", "error")),
    ),
}

_ALL_METRIC_HELP = {**METRIC_HELP, **_COUNTER_HELP}


class _Histogram:
    def __init__(self) -> None:
        self.bucket_counts = [0 for _ in range(len(BUCKETS) + 1)]
        self.total = 0.0
        self.n = 0

    def observe(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            return
        self.total += value
        self.n += 1
        for index, bound in enumerate(BUCKETS):
            if value <= bound:
                self.bucket_counts[index] += 1
                return
        self.bucket_counts[-1] += 1


class _Registry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.histograms: dict[str, _Histogram] = {}
        self.errors: dict[str, int] = {}
        self.gauges: dict[str, float] = {}
        self.counters: dict[str, dict[tuple[str, ...], int]] = {}

    def observe(self, stage: str, seconds: float) -> None:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage}")
        with self._lock:
            histogram = self.histograms.get(stage)
            if histogram is None:
                histogram = _Histogram()
                self.histograms[stage] = histogram
            histogram.observe(seconds)

    def error(self, stage: str) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError(f"unknown stage {stage}")
        with self._lock:
            self.errors[stage] = self.errors.get(stage, 0) + 1

    def gauge(self, name: str, value: float) -> None:
        if name not in METRIC_HELP or not math.isfinite(value):
            raise ValueError(f"cannot record gauge {name}")
        with self._lock:
            self.gauges[name] = float(value)

    def counter(self, name: str, values: tuple[str, ...]) -> None:
        labels = _COUNTER_LABELS.get(name)
        if labels is None or len(values) != len(labels):
            raise ValueError(f"cannot record counter {name}")
        for (label, allowed), value in zip(labels, values, strict=True):
            if value not in allowed:
                raise ValueError(f"invalid {label} for {name}")
        with self._lock:
            series = self.counters.setdefault(name, {})
            series[values] = series.get(values, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            histograms = {
                stage: {
                    "bucket_counts": list(histogram.bucket_counts),
                    "total": histogram.total,
                    "n": histogram.n,
                }
                for stage, histogram in self.histograms.items()
            }
            return {
                "histograms": histograms,
                "errors": dict(self.errors),
                "gauges": dict(self.gauges),
                "counters": {name: dict(series) for name, series in self.counters.items()},
            }

    def reset(self) -> None:
        with self._lock:
            self.histograms.clear()
            self.errors.clear()
            self.gauges.clear()
            self.counters.clear()


REGISTRY = _Registry()


def reset_metrics() -> None:
    REGISTRY.reset()


def record_latency(stage: str, seconds: float) -> None:
    if not observe_enabled():
        return
    REGISTRY.observe(stage, seconds)


def record_error(stage: str) -> None:
    if not observe_enabled():
        return
    REGISTRY.error(stage)


def set_data_freshness(seconds: float) -> None:
    if not observe_enabled():
        return
    REGISTRY.gauge("dipcatcher_data_freshness_seconds", seconds)


def set_simulation_state(*, pnl: float, gross_exposure: float, net_exposure: float) -> None:
    """Record simulation gauges. The values are not research results."""
    if not observe_enabled():
        return
    REGISTRY.gauge("dipcatcher_simulation_pnl", pnl)
    REGISTRY.gauge("dipcatcher_simulation_gross_exposure", gross_exposure)
    REGISTRY.gauge("dipcatcher_simulation_net_exposure", net_exposure)


def _record_live_counter(name: str, values: tuple[str, ...]) -> None:
    if not observe_enabled():
        return
    REGISTRY.counter(name, values)


def record_live_order_submit(*, venue: str) -> None:
    _record_live_counter("dipcatcher_live_order_submits_total", (venue,))


def record_live_order_result(*, venue: str, status: str) -> None:
    _record_live_counter("dipcatcher_live_order_results_total", (venue, status))


def record_live_fill_record(*, venue: str, verdict: str) -> None:
    _record_live_counter("dipcatcher_live_fill_records_total", (venue, verdict))


def record_live_pretrade_denial(*, venue: str, kind: str, primary_reason: str) -> None:
    _record_live_counter(
        "dipcatcher_live_pretrade_denials_total",
        (venue, kind, primary_reason),
    )


def record_live_kill_switch_transition(*, venue: str, state: str) -> None:
    _record_live_counter("dipcatcher_live_kill_switch_transitions_total", (venue, state))


def record_live_order_recovery(*, venue: str, outcome: str) -> None:
    _record_live_counter("dipcatcher_live_order_recovery_attempts_total", (venue, outcome))


def histogram_quantile_upper(stage: str, quantile: float) -> float:
    """Bucket upper bound covering ``quantile``. ``nan`` when the stage has no samples.

    This is the histogram bucket bound, not an interpolated quantile.
    """
    snap = REGISTRY.snapshot()["histograms"].get(stage)
    if not snap or snap["n"] == 0:
        return math.nan
    target = quantile * snap["n"]
    cumulative = 0
    for index, bound in enumerate(BUCKETS):
        cumulative += snap["bucket_counts"][index]
        if cumulative >= target:
            return bound
    return math.inf


def register_ops_sampler(sampler: _OpsSampler) -> None:
    """Register the lab scrape sampler. Values are read on each exposition."""
    global _OPS_SAMPLER
    _OPS_SAMPLER = sampler


def clear_ops_sampler() -> None:
    """Drop the lab scrape sampler. Used when the process stops and in tests."""
    global _OPS_SAMPLER
    _OPS_SAMPLER = None


def _append_operator_gauges(lines: list[str]) -> None:
    """Add readiness, drain, and kill series when the lab process registered them.

    The series are operator signals, not research scores. A missing sampler
    leaves the exposition unchanged. The kill label is the reader code.
    """
    sampler = _OPS_SAMPLER
    if sampler is None:
        return
    ready, draining, blocking, code = sampler()
    if not _unit_gauge(ready) or not _unit_gauge(draining) or not _unit_gauge(blocking):
        return
    safe = code if _OPS_CODE.fullmatch(code) else "kill_switch_invalid"
    _emit_gauge(lines, "dipcatcher_ready", ready)
    _emit_gauge(lines, "dipcatcher_draining", draining)
    _emit_gauge(lines, "dipcatcher_kill_switch_blocking", blocking)
    _help(lines, "dipcatcher_kill_switch_info", "gauge")
    lines.append(f'dipcatcher_kill_switch_info{{code="{safe}"}} 1')


def _unit_gauge(value: object) -> bool:
    return isinstance(value, float) and value in {0.0, 1.0}


def _emit_gauge(lines: list[str], name: str, value: float) -> None:
    _help(lines, name, "gauge")
    lines.append(f"{name} {_num(value)}")


def render_prometheus() -> str:
    """Prometheus text exposition 0.0.4 for the in-process registry."""
    snap = REGISTRY.snapshot()
    lines: list[str] = []
    if snap["histograms"]:
        _help(lines, "dipcatcher_stage_latency_seconds", "histogram")
    for stage in STAGES:
        histogram = snap["histograms"].get(stage)
        if histogram is None:
            continue
        cumulative = 0
        for index, bound in enumerate(BUCKETS):
            cumulative += histogram["bucket_counts"][index]
            lines.append(
                f'dipcatcher_stage_latency_seconds_bucket{{stage="{stage}",le="{_num(bound)}"}} {cumulative}'
            )
        cumulative += histogram["bucket_counts"][-1]
        lines.append(
            f'dipcatcher_stage_latency_seconds_bucket{{stage="{stage}",le="+Inf"}} {cumulative}'
        )
        lines.append(
            f'dipcatcher_stage_latency_seconds_sum{{stage="{stage}"}} {_num(histogram["total"])}'
        )
        lines.append(f'dipcatcher_stage_latency_seconds_count{{stage="{stage}"}} {histogram["n"]}')
    if snap["errors"]:
        _help(lines, "dipcatcher_errors_total", "counter")
    for stage in (*STAGES, "otel_export"):
        if stage in snap["errors"]:
            lines.append(f'dipcatcher_errors_total{{stage="{stage}"}} {snap["errors"][stage]}')
    for name in (
        "dipcatcher_data_freshness_seconds",
        "dipcatcher_simulation_pnl",
        "dipcatcher_simulation_gross_exposure",
        "dipcatcher_simulation_net_exposure",
    ):
        if name in snap["gauges"]:
            _help(lines, name, "gauge")
            lines.append(f"{name} {_num(snap['gauges'][name])}")
    _append_operator_gauges(lines)
    for name, labels in _COUNTER_LABELS.items():
        series = snap["counters"].get(name)
        if not series:
            continue
        _help(lines, name, "counter")
        for values in sorted(series):
            label_text = ",".join(
                f'{label}="{value}"' for (label, _), value in zip(labels, values, strict=True)
            )
            lines.append(f"{name}{{{label_text}}} {series[values]}")
    text = "\n".join(lines)
    if text:
        text += "\n"
    return text


def _help(lines: list[str], name: str, metric_type: str) -> None:
    lines.append(f"# HELP {name} {_ALL_METRIC_HELP[name]}")
    lines.append(f"# TYPE {name} {metric_type}")


def _num(value: float) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == 0:
        return "0"
    return format(value, ".12g")
