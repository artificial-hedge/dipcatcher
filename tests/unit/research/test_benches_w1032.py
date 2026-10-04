"""Wave-1032 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1032 import (
    bench_aerodynamics_family,
    bench_airfoil_theory_family,
    bench_flight_dynamics_family,
    bench_orbital_mechanics2_family,
    bench_propulsion_family,
    bench_spacecraft_design_family,
)

_FAMILY_BENCHES = [
    bench_aerodynamics_family,
    bench_propulsion_family,
    bench_orbital_mechanics2_family,
    bench_flight_dynamics_family,
    bench_spacecraft_design_family,
    bench_airfoil_theory_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
