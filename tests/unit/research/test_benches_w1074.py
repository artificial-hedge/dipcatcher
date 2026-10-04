"""Wave-1074 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1074 import (
    bench_baking_science_family,
    bench_culinary_arts_family,
    bench_fermentation_science_family,
    bench_flavor_science_family,
    bench_food_studies_family,
    bench_gastronomy_family,
)

_FAMILY_BENCHES = [
    bench_culinary_arts_family,
    bench_gastronomy_family,
    bench_food_studies_family,
    bench_baking_science_family,
    bench_flavor_science_family,
    bench_fermentation_science_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
