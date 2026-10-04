"""Wave-1155 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1155 import (
    bench_atmospheric_science_family,
    bench_earth_system_science_family,
    bench_environmental_science_2_family,
    bench_hydrology_3_family,
    bench_oceanography_2_family,
    bench_soil_science_2_family,
)

_FAMILY_BENCHES = [
    bench_earth_system_science_family,
    bench_oceanography_2_family,
    bench_atmospheric_science_family,
    bench_environmental_science_2_family,
    bench_soil_science_2_family,
    bench_hydrology_3_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
