"""Wave-933 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w933 import (
    bench_bregman_proj_family,
    bench_conjugate_fn_family,
    bench_fenchel_dual_family,
    bench_moreau_env_family,
    bench_proximal_map_family,
    bench_subgradient_proj_family,
)

_FAMILY_BENCHES = [
    bench_subgradient_proj_family,
    bench_proximal_map_family,
    bench_fenchel_dual_family,
    bench_moreau_env_family,
    bench_bregman_proj_family,
    bench_conjugate_fn_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
