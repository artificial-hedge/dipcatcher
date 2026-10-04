"""Wave-1144 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1144 import (
    bench_asteroid_science_family,
    bench_astrophotonics_family,
    bench_comet_science_family,
    bench_grav_waves_2_family,
    bench_planetology_family,
    bench_space_weather_family,
)

_FAMILY_BENCHES = [
    bench_space_weather_family,
    bench_planetology_family,
    bench_asteroid_science_family,
    bench_comet_science_family,
    bench_astrophotonics_family,
    bench_grav_waves_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
