"""Wave-1215 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1215 import (
    bench_adult_congenital_family,
    bench_cardiac_surgery_family,
    bench_structural_heart_family,
    bench_thoracic_surgery_family,
    bench_transplant_cardiology_family,
    bench_vascular_surgery_family,
)

_FAMILY_BENCHES = [
    bench_vascular_surgery_family,
    bench_cardiac_surgery_family,
    bench_thoracic_surgery_family,
    bench_transplant_cardiology_family,
    bench_structural_heart_family,
    bench_adult_congenital_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
