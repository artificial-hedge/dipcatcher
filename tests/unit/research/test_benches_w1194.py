"""Wave-1194 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1194 import (
    bench_genetic_counseling_family,
    bench_lactation_consulting_family,
    bench_perfusion_technology_family,
    bench_podiatric_medicine_family,
    bench_radiation_therapy_family,
    bench_respiratory_therapy_family,
)

_FAMILY_BENCHES = [
    bench_genetic_counseling_family,
    bench_lactation_consulting_family,
    bench_podiatric_medicine_family,
    bench_respiratory_therapy_family,
    bench_perfusion_technology_family,
    bench_radiation_therapy_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
