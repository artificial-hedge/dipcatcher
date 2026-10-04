"""Wave-1008 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1008 import (
    bench_carnot_cycle_family,
    bench_critical_phenomena_family,
    bench_entropy_production_family,
    bench_fluctuation_dissipation_family,
    bench_maxwell_relations_family,
    bench_phase_transitions_family,
)

_FAMILY_BENCHES = [
    bench_carnot_cycle_family,
    bench_maxwell_relations_family,
    bench_phase_transitions_family,
    bench_critical_phenomena_family,
    bench_fluctuation_dissipation_family,
    bench_entropy_production_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
