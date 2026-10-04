"""Wave-1154 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1154 import (
    bench_electromagnetism_family,
    bench_nuclear_physics_2_family,
    bench_optics_4_family,
    bench_particle_physics_family,
    bench_quantum_physics_family,
    bench_relativity_3_family,
)

_FAMILY_BENCHES = [
    bench_electromagnetism_family,
    bench_optics_4_family,
    bench_nuclear_physics_2_family,
    bench_particle_physics_family,
    bench_quantum_physics_family,
    bench_relativity_3_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
