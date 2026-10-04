"""Wave-1055 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1055 import (
    bench_archaeology_family,
    bench_cultural_anthropology_family,
    bench_ethnography_family,
    bench_linguistic_anthropology_family,
    bench_physical_anthropology_family,
    bench_primatology_family,
)

_FAMILY_BENCHES = [
    bench_physical_anthropology_family,
    bench_cultural_anthropology_family,
    bench_archaeology_family,
    bench_linguistic_anthropology_family,
    bench_primatology_family,
    bench_ethnography_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
