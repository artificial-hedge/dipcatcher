"""Wave-912 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w912 import (
    bench_hilbert_curve_family,
    bench_morton_order_family,
    bench_octree_index_family,
    bench_range_tree_family,
    bench_rstar_tree_family,
    bench_z_curve_family,
)

_FAMILY_BENCHES = [
    bench_octree_index_family,
    bench_range_tree_family,
    bench_hilbert_curve_family,
    bench_z_curve_family,
    bench_morton_order_family,
    bench_rstar_tree_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
