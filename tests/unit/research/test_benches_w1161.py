"""Wave-1161 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1161 import (
    bench_dentistry_3_family,
    bench_medicine_7_family,
    bench_nursing_2_family,
    bench_pharmacy_2_family,
    bench_public_health_2_family,
    bench_veterinary_medicine_2_family,
)

_FAMILY_BENCHES = [
    bench_medicine_7_family,
    bench_dentistry_3_family,
    bench_nursing_2_family,
    bench_public_health_2_family,
    bench_veterinary_medicine_2_family,
    bench_pharmacy_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
