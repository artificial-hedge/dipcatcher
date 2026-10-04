"""Wave-1149 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1149 import (
    bench_dermatology_2_family,
    bench_hematology_2_family,
    bench_hepatology_2_family,
    bench_nephrology_2_family,
    bench_pulmonology_2_family,
    bench_toxicology_2_family,
)

_FAMILY_BENCHES = [
    bench_toxicology_2_family,
    bench_dermatology_2_family,
    bench_hematology_2_family,
    bench_pulmonology_2_family,
    bench_nephrology_2_family,
    bench_hepatology_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
