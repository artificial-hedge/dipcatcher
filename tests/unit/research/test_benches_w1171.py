"""Wave-1171 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1171 import (
    bench_demography_2_family,
    bench_geography_2_family,
    bench_gis_science_2_family,
    bench_land_use_family,
    bench_regional_science_family,
    bench_urbanization_family,
)

_FAMILY_BENCHES = [
    bench_geography_2_family,
    bench_regional_science_family,
    bench_demography_2_family,
    bench_urbanization_family,
    bench_land_use_family,
    bench_gis_science_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
