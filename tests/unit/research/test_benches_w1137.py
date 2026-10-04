"""Wave-1137 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1137 import (
    bench_adult_education_family,
    bench_bilingual_education_family,
    bench_early_childhood_education_family,
    bench_educational_leadership_family,
    bench_gifted_education_family,
    bench_instructional_design_family,
)

_FAMILY_BENCHES = [
    bench_early_childhood_education_family,
    bench_bilingual_education_family,
    bench_gifted_education_family,
    bench_adult_education_family,
    bench_instructional_design_family,
    bench_educational_leadership_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
