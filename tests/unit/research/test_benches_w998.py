"""Wave-998 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w998 import (
    bench_fujita_exponent_family,
    bench_matched_asymptotic_pde_family,
    bench_regularity_critical_family,
    bench_self_similar_blowup_family,
    bench_semilinear_heat_family,
    bench_singularity_formation_family,
)

_FAMILY_BENCHES = [
    bench_semilinear_heat_family,
    bench_fujita_exponent_family,
    bench_singularity_formation_family,
    bench_matched_asymptotic_pde_family,
    bench_self_similar_blowup_family,
    bench_regularity_critical_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
