"""Wave-967 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w967 import (
    bench_crossed_product_family,
    bench_cstar_dynamics_family,
    bench_kirchberg_absorb_family,
    bench_rokhlin_action_family,
    bench_taf_dim_family,
    bench_z_stability_family,
)

_FAMILY_BENCHES = [
    bench_cstar_dynamics_family,
    bench_crossed_product_family,
    bench_rokhlin_action_family,
    bench_kirchberg_absorb_family,
    bench_taf_dim_family,
    bench_z_stability_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
