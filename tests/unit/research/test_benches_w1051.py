"""Wave-1051 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1051 import (
    bench_biostatistics_2_family,
    bench_epidemiology_2_family,
    bench_global_health_family,
    bench_health_policy_family,
    bench_occupational_health_family,
    bench_preventive_medicine_family,
)

_FAMILY_BENCHES = [
    bench_epidemiology_2_family,
    bench_biostatistics_2_family,
    bench_health_policy_family,
    bench_global_health_family,
    bench_occupational_health_family,
    bench_preventive_medicine_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
