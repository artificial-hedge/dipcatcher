"""Wave-982 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w982 import (
    bench_boundary_regular_family,
    bench_capacitary_pot_family,
    bench_dirichlet_problem_family,
    bench_energy_principle_family,
    bench_equilibrium_measure_family,
    bench_thin_set_family,
)

_FAMILY_BENCHES = [
    bench_dirichlet_problem_family,
    bench_energy_principle_family,
    bench_equilibrium_measure_family,
    bench_thin_set_family,
    bench_boundary_regular_family,
    bench_capacitary_pot_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
