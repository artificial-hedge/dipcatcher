"""Wave-1017 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w1017 import (
    bench_navier_cauchy_family,
    bench_plasticity_family,
    bench_poroelasticity_family,
    bench_rheology_family,
    bench_stress_tensor_family,
    bench_viscoelasticity_family,
)

_FAMILY_BENCHES = [
    bench_navier_cauchy_family,
    bench_stress_tensor_family,
    bench_rheology_family,
    bench_viscoelasticity_family,
    bench_plasticity_family,
    bench_poroelasticity_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
