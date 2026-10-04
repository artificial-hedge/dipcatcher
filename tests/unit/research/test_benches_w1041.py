"""Wave-1041 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1041 import (
    bench_coastal_engineering_family,
    bench_marine_propulsion_family,
    bench_naval_architecture_family,
    bench_ocean_waves_family,
    bench_offshore_engineering_family,
    bench_submarine_systems_family,
)

_FAMILY_BENCHES = [
    bench_naval_architecture_family,
    bench_offshore_engineering_family,
    bench_marine_propulsion_family,
    bench_ocean_waves_family,
    bench_coastal_engineering_family,
    bench_submarine_systems_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
