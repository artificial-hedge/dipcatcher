"""Wave-1026 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1026 import (
    bench_atmospheric_chem_family,
    bench_carbon_cycle_family,
    bench_climate_model_family,
    bench_ecosystem_model_family,
    bench_hydrology_family,
    bench_ocean_circulation_family,
)

_FAMILY_BENCHES = [
    bench_climate_model_family,
    bench_ocean_circulation_family,
    bench_atmospheric_chem_family,
    bench_hydrology_family,
    bench_carbon_cycle_family,
    bench_ecosystem_model_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
