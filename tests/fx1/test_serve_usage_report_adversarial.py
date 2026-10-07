"""SYNTHETIC adversarial probes for ``fx1.serve.usage_report``.

Billing-honesty edges of the aggregate: the unbillable sieve (bools,
strings, floats never sum), verbatim negative/zero counters, inclusive
time windows, ``(none)`` bucket collisions, NaN record handling, and the
closed extra=forbid envelope. All fixtures are synthetic records, never
market evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError

from fx1.serve.usage_report import UsageBucket, UsageReport, aggregate_usage


@dataclass(frozen=True)
class _Rec:
    """Minimal record satisfying the ``_Record`` protocol."""

    backend: str = "byok"
    model: str | None = "m"
    ok: bool = True
    latency_ms: float = 1.0
    at: float = 100.0
    usage: dict[str, Any] | None = None
    key_id: str | None = None


def _agg(
    records: list[_Rec],
    *,
    since: float | None = None,
    until: float | None = None,
    backend: str | None = None,
    model: str | None = None,
    key_id: str | None = None,
) -> UsageReport:
    return aggregate_usage(
        records,
        cap=64,
        dropped=0,
        since=since,
        until=until,
        backend=backend,
        model=model,
        key_id=key_id,
    )


def test_unbillable_claims_are_sieved_everywhere() -> None:
    """Bools, strings and floats in a usage dict never enter billable sums —
    at totals level and inside every split bucket."""
    usage = {
        "prompt_tokens": True,  # bool is not billable
        "completion_tokens": "9",  # string is not billable
        "total_tokens": 4.5,  # float is not billable
        "provider_custom": 7,  # non-canonical int survives verbatim
    }
    report = _agg([_Rec(usage=usage)])
    totals = report.totals
    assert totals.usage_reported == 1
    assert totals.prompt_tokens == 0
    assert totals.completion_tokens == 0
    assert totals.total_tokens == 0
    assert totals.other_usage == {"provider_custom": 7}
    assert report.by_backend["byok"].other_usage == {"provider_custom": 7}
    assert report.by_model["m"].other_usage == {"provider_custom": 7}
    assert report.by_key["(none)"].other_usage == {"provider_custom": 7}


def test_negative_and_zero_claims_sum_verbatim() -> None:
    """Provider-reported negatives are summed verbatim — the aggregate never
    silently corrects a counter; charge-side clamping lives elsewhere."""
    usage = {"prompt_tokens": -5, "completion_tokens": 0, "weird": -3}
    report = _agg([_Rec(usage=usage)])
    assert report.totals.prompt_tokens == -5
    assert report.totals.completion_tokens == 0
    assert report.totals.other_usage == {"weird": -3}


def test_window_bounds_are_inclusive() -> None:
    edge_lo = _Rec(at=100.0)
    edge_hi = _Rec(at=200.0)
    outside = _Rec(at=300.0)
    report = _agg([edge_lo, edge_hi, outside], since=100.0, until=200.0)
    assert report.records_seen == 2
    assert _agg([edge_lo, edge_hi], since=100.001).records_seen == 1
    assert _agg([edge_lo, edge_hi], until=199.999).records_seen == 1


def test_nan_timestamp_drops_out_of_windowed_reports_only() -> None:
    """NaN ``at`` comparisons are always false: a corrupt-timestamp record is
    invisible in windowed queries yet still counted unfiltered — the
    divergence is declared, never hidden."""
    nan_rec = _Rec(at=float("nan"))
    assert _agg([nan_rec]).records_seen == 1
    assert _agg([nan_rec], since=0.0).records_seen == 0
    assert _agg([nan_rec], until=1e18).records_seen == 0


def test_nan_latency_poisoning_is_explicit_in_mean() -> None:
    """A NaN latency propagates to mean_latency_ms rather than being silently
    dropped — a corrupt record visibly corrupts the statistic."""
    report = _agg([_Rec(latency_ms=1.0), _Rec(latency_ms=float("nan"))])
    assert math.isnan(report.totals.mean_latency_ms or 0.0)


def test_empty_usage_dict_counts_as_reported_with_zero_tokens() -> None:
    """``usage={}`` is a real report event: usage_reported ticks up while all
    token sums stay zero — reported-vs-totaled stays distinguishable."""
    report = _agg([_Rec(usage={})])
    assert report.totals.usage_reported == 1
    assert report.totals.prompt_tokens == 0
    assert report.totals.other_usage == {}


def test_absent_usage_never_fabricates() -> None:
    report = _agg([_Rec(usage=None)])
    assert report.totals.usage_reported == 0
    assert report.totals.total_tokens == 0


def test_model_none_buckets_as_label_and_collides_with_literal() -> None:
    """model=None lands in the ``(none)`` bucket; a provider literally named
    ``(none)`` merges into it — the label collision is measured, not hidden."""
    report = _agg([_Rec(model=None), _Rec(model="(none)")])
    assert list(report.by_model) == ["(none)"]
    assert report.by_model["(none)"].requests == 2


def test_filters_compose_as_conjunction() -> None:
    recs = [
        _Rec(model="a", key_id="k1", at=10.0),
        _Rec(model="a", key_id="k2", at=10.0),
        _Rec(model="b", key_id="k1", at=10.0),
        _Rec(model="a", key_id="k1", at=999.0),
    ]
    report = _agg(recs, model="a", key_id="k1", until=100.0)
    assert report.records_seen == 1
    assert report.model == "a" and report.key_id == "k1" and report.until == 100.0


def test_backend_param_is_echoed_not_applied() -> None:
    """backend filtering is the caller's job (``log.all(backend)``); the
    parameter is echoed in the report, never silently re-filtered here."""
    recs = [_Rec(backend="b1"), _Rec(backend="b2")]
    report = _agg(recs, backend="b1")
    assert report.backend == "b1"
    assert report.records_seen == 2  # both records still folded
    assert sorted(report.by_backend) == ["b1", "b2"]


def test_forbidden_headline_named_counters_pass_through_verbatim() -> None:
    """A provider counter literally named 'pnl'/'nav' is summed verbatim into
    other_usage — accounting honesty beats key-name cosmetics (the sealed
    receipt layer flags such keys at verify, tested in ops_receipt probes)."""
    usage = {"pnl": 3, "nav": 2, "prompt_tokens": 5}
    report = _agg([_Rec(usage=usage)])
    assert report.totals.other_usage == {"nav": 2, "pnl": 3}
    assert report.totals.prompt_tokens == 5


def test_split_ordering_is_sorted_and_deterministic() -> None:
    recs = [
        _Rec(backend="z", model="z", key_id="z"),
        _Rec(backend="a", model="a", key_id="a"),
        _Rec(backend="m", model="m", key_id="m"),
    ]
    report = _agg(recs)
    assert list(report.by_backend) == ["a", "m", "z"]
    assert list(report.by_model) == ["a", "m", "z"]
    assert list(report.by_key) == ["a", "m", "z"]


def test_report_models_reject_extra_fields() -> None:
    """The usage envelope is closed — smuggled fields fail validation."""
    bucket = {
        "requests": 0,
        "ok": 0,
        "errors": 0,
        "usage_reported": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "other_usage": {},
        "mean_latency_ms": None,
    }
    with pytest.raises(ValidationError):
        UsageReport.model_validate(
            {
                "generated_at": 0.0,
                "since": None,
                "until": None,
                "backend": None,
                "model": None,
                "key_id": None,
                "records_seen": 0,
                "records_dropped": 0,
                "ring_cap": 0,
                "totals": bucket,
                "by_backend": {},
                "by_model": {},
                "by_key": {},
                "smuggled": 1,
            }
        )
    with pytest.raises(ValidationError):
        UsageBucket.model_validate({**bucket, "smuggled": 1})


def test_empty_window_and_serializable_output() -> None:
    """An empty aggregate is honest: zeroed counters, null mean, strict-JSON
    serializable."""
    report = _agg([])
    assert report.totals.requests == 0 and report.totals.mean_latency_ms is None
    blob = report.model_dump_json()
    assert "NaN" not in blob and "Infinity" not in blob
    assert report.records_seen == 0 and report.records_dropped == 0
    assert report.ring_cap == 64
