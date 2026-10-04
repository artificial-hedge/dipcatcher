"""Wave-1046 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1046 import (
    bench_atmospheric_dynamics_family,
    bench_climate_dynamics_family,
    bench_cloud_physics_family,
    bench_mesoscale_meteorology_family,
    bench_numerical_weather_family,
    bench_synoptic_meteorology_family,
)

_FAMILY_BENCHES = [
    bench_atmospheric_dynamics_family,
    bench_synoptic_meteorology_family,
    bench_cloud_physics_family,
    bench_numerical_weather_family,
    bench_mesoscale_meteorology_family,
    bench_climate_dynamics_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
