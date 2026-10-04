"""Wave-1165 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1165 import (
    bench_agriculture_2_family,
    bench_fisheries_2_family,
    bench_food_science_2_family,
    bench_forestry_2_family,
    bench_horticulture_2_family,
    bench_veterinary_science_2_family,
)

_FAMILY_BENCHES = [
    bench_agriculture_2_family,
    bench_food_science_2_family,
    bench_forestry_2_family,
    bench_fisheries_2_family,
    bench_horticulture_2_family,
    bench_veterinary_science_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
