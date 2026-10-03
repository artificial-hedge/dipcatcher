"""Wave-908 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w908 import (
    bench_fenwick_tree_family,
    bench_merge_sort_tree_family,
    bench_segment_tree_family,
    bench_sparse_table_family,
    bench_sqrt_decomp_family,
    bench_wavelet_tree_family,
)

_FAMILY_BENCHES = [
    bench_segment_tree_family,
    bench_fenwick_tree_family,
    bench_sparse_table_family,
    bench_sqrt_decomp_family,
    bench_wavelet_tree_family,
    bench_merge_sort_tree_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
