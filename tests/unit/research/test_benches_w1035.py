"""Wave-1035 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1035 import (
    bench_isotope_production_family,
    bench_nuclear_fuel_cycle_family,
    bench_nuclear_safety_family,
    bench_radiation_protection_family,
    bench_reactor_physics_family,
    bench_thermal_hydraulics_family,
)

_FAMILY_BENCHES = [
    bench_reactor_physics_family,
    bench_radiation_protection_family,
    bench_nuclear_fuel_cycle_family,
    bench_thermal_hydraulics_family,
    bench_nuclear_safety_family,
    bench_isotope_production_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
