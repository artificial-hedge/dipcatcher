"""Wave-1129 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1129 import (
    bench_african_philosophy_family,
    bench_bioethics_family,
    bench_environmental_philosophy_family,
    bench_feminist_philosophy_family,
    bench_philosophy_of_education_family,
    bench_philosophy_of_medicine_family,
)

_FAMILY_BENCHES = [
    bench_bioethics_family,
    bench_philosophy_of_education_family,
    bench_feminist_philosophy_family,
    bench_african_philosophy_family,
    bench_environmental_philosophy_family,
    bench_philosophy_of_medicine_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
