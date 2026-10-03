"""Wave-1000 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1000 import (
    bench_contact_mechanics_family,
    bench_fracture_mechanics_family,
    bench_homogenized_elasticity_family,
    bench_kirchhoff_plate_family,
    bench_mindlin_reissner_family,
    bench_navier_elasticity_family,
)

_FAMILY_BENCHES = [
    bench_navier_elasticity_family,
    bench_kirchhoff_plate_family,
    bench_mindlin_reissner_family,
    bench_contact_mechanics_family,
    bench_fracture_mechanics_family,
    bench_homogenized_elasticity_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
