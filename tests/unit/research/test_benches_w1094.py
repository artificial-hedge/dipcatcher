"""Wave-1094 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1094 import (
    bench_dermatology_family,
    bench_neurology_family,
    bench_oncology_family,
    bench_orthopedics_family,
    bench_psychiatry_family,
    bench_radiology_family,
)

_FAMILY_BENCHES = [
    bench_oncology_family,
    bench_neurology_family,
    bench_dermatology_family,
    bench_orthopedics_family,
    bench_psychiatry_family,
    bench_radiology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
