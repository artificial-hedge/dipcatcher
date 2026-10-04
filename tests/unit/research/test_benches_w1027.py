"""Wave-1027 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1027 import (
    bench_ceramics_family,
    bench_crystal_structure_family,
    bench_metallurgy_family,
    bench_nanomaterials_family,
    bench_polymer_physics_family,
    bench_superconductivity_family,
)

_FAMILY_BENCHES = [
    bench_crystal_structure_family,
    bench_polymer_physics_family,
    bench_metallurgy_family,
    bench_ceramics_family,
    bench_nanomaterials_family,
    bench_superconductivity_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
