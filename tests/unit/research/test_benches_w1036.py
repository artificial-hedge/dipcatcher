"""Wave-1036 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1036 import (
    bench_drilling_engineering_family,
    bench_enhanced_recovery_family,
    bench_formation_evaluation_family,
    bench_production_engineering_family,
    bench_reservoir_engineering_family,
    bench_well_testing_family,
)

_FAMILY_BENCHES = [
    bench_reservoir_engineering_family,
    bench_drilling_engineering_family,
    bench_production_engineering_family,
    bench_formation_evaluation_family,
    bench_well_testing_family,
    bench_enhanced_recovery_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
