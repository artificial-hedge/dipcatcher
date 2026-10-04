"""Wave-904 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w904 import (
    bench_aa_tree_family,
    bench_avl_tree_family,
    bench_red_black_tree_family,
    bench_scapegoat_tree_family,
    bench_splay_tree_family,
    bench_treap_family,
)

_FAMILY_BENCHES = [
    bench_avl_tree_family,
    bench_red_black_tree_family,
    bench_splay_tree_family,
    bench_treap_family,
    bench_scapegoat_tree_family,
    bench_aa_tree_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
