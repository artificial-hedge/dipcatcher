"""Wave-1105 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1105 import (
    bench_bounded_rationality_family,
    bench_experimental_economics_family,
    bench_financial_behavior_family,
    bench_neuroeconomics_family,
    bench_nudge_theory_family,
    bench_prospect_theory_family,
)

_FAMILY_BENCHES = [
    bench_prospect_theory_family,
    bench_bounded_rationality_family,
    bench_nudge_theory_family,
    bench_neuroeconomics_family,
    bench_experimental_economics_family,
    bench_financial_behavior_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
