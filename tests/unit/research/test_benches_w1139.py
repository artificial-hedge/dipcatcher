"""Wave-1139 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1139 import (
    bench_dentistry_2_family,
    bench_dietetics_family,
    bench_occupational_therapy_family,
    bench_optometry_family,
    bench_physiotherapy_family,
    bench_podiatry_family,
)

_FAMILY_BENCHES = [
    bench_optometry_family,
    bench_dentistry_2_family,
    bench_podiatry_family,
    bench_dietetics_family,
    bench_physiotherapy_family,
    bench_occupational_therapy_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
