"""Wave-1037 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1037 import (
    bench_agronomy_family,
    bench_animal_science_family,
    bench_crop_science_family,
    bench_horticulture_family,
    bench_pest_management_family,
    bench_soil_science_family,
)

_FAMILY_BENCHES = [
    bench_crop_science_family,
    bench_soil_science_family,
    bench_agronomy_family,
    bench_animal_science_family,
    bench_horticulture_family,
    bench_pest_management_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
