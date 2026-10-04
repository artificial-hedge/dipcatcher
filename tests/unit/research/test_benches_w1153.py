"""Wave-1153 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1153 import (
    bench_astrophysics_3_family,
    bench_cosmology_3_family,
    bench_geophysics_3_family,
    bench_mechanics_family,
    bench_physics_6_family,
    bench_thermodynamics_3_family,
)

_FAMILY_BENCHES = [
    bench_physics_6_family,
    bench_astrophysics_3_family,
    bench_cosmology_3_family,
    bench_geophysics_3_family,
    bench_mechanics_family,
    bench_thermodynamics_3_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
