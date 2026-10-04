"""Wave-1108 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1108 import (
    bench_cultural_sociology_family,
    bench_environmental_sociology_family,
    bench_industrial_sociology_family,
    bench_political_sociology_family,
    bench_sociology_of_education_family,
    bench_sociology_of_religion_family,
)

_FAMILY_BENCHES = [
    bench_industrial_sociology_family,
    bench_political_sociology_family,
    bench_sociology_of_education_family,
    bench_sociology_of_religion_family,
    bench_environmental_sociology_family,
    bench_cultural_sociology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
