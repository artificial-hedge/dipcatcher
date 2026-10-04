"""Wave-1121 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1121 import (
    bench_economic_geography_family,
    bench_gis_science_family,
    bench_health_geography_family,
    bench_political_geography_family,
    bench_population_geography_family,
    bench_regional_geography_family,
)

_FAMILY_BENCHES = [
    bench_regional_geography_family,
    bench_health_geography_family,
    bench_population_geography_family,
    bench_economic_geography_family,
    bench_political_geography_family,
    bench_gis_science_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
