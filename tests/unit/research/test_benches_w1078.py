"""Wave-1078 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1078 import (
    bench_choreography_family,
    bench_dance_studies_family,
    bench_dramaturgy_family,
    bench_performance_theory_family,
    bench_stage_design_family,
    bench_theater_studies_family,
)

_FAMILY_BENCHES = [
    bench_theater_studies_family,
    bench_dance_studies_family,
    bench_performance_theory_family,
    bench_dramaturgy_family,
    bench_choreography_family,
    bench_stage_design_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
