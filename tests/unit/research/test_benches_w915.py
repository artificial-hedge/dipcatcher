"""Wave-915 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w915 import (
    bench_bvp_eigen_family,
    bench_continuation_bvp_family,
    bench_fusion_tree_family,
    bench_loser_tree_family,
    bench_robbins_bvp_family,
    bench_superposition_bvp_family,
)

_FAMILY_BENCHES = [
    bench_superposition_bvp_family,
    bench_continuation_bvp_family,
    bench_robbins_bvp_family,
    bench_bvp_eigen_family,
    bench_loser_tree_family,
    bench_fusion_tree_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
