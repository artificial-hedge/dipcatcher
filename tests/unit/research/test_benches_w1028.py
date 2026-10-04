"""Wave-1028 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1028 import (
    bench_fluid_dynamics2_family,
    bench_heat_exchanger_family,
    bench_process_control_family,
    bench_reaction_kinetics_family,
    bench_separation_proc_family,
    bench_thermo_props_family,
)

_FAMILY_BENCHES = [
    bench_reaction_kinetics_family,
    bench_thermo_props_family,
    bench_separation_proc_family,
    bench_heat_exchanger_family,
    bench_fluid_dynamics2_family,
    bench_process_control_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
