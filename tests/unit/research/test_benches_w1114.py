"""Wave-1114 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1114 import (
    bench_classical_mechanics_family,
    bench_condensed_matter_2_family,
    bench_nuclear_physics_family,
    bench_plasma_physics_family,
    bench_quantum_mechanics_2_family,
    bench_statistical_mechanics_2_family,
)

_FAMILY_BENCHES = [
    bench_classical_mechanics_family,
    bench_quantum_mechanics_2_family,
    bench_statistical_mechanics_2_family,
    bench_nuclear_physics_family,
    bench_plasma_physics_family,
    bench_condensed_matter_2_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
