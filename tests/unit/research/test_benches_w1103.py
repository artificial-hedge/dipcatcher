"""Wave-1103 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1103 import (
    bench_boundary_layer_meteorology_family,
    bench_micrometeorology_family,
    bench_polar_meteorology_family,
    bench_radar_meteorology_family,
    bench_severe_weather_family,
    bench_tropical_meteorology_family,
)

_FAMILY_BENCHES = [
    bench_severe_weather_family,
    bench_boundary_layer_meteorology_family,
    bench_radar_meteorology_family,
    bench_tropical_meteorology_family,
    bench_polar_meteorology_family,
    bench_micrometeorology_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
