"""Wave-1091 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1091 import (
    bench_comparative_education_family,
    bench_distance_learning_family,
    bench_higher_education_family,
    bench_literacy_studies_family,
    bench_special_education_family,
    bench_vocational_education_family,
)

_FAMILY_BENCHES = [
    bench_higher_education_family,
    bench_vocational_education_family,
    bench_special_education_family,
    bench_comparative_education_family,
    bench_literacy_studies_family,
    bench_distance_learning_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
