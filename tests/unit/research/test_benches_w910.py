"""Wave-910 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w910 import (
    bench_b_plus_tree_family,
    bench_b_star_tree_family,
    bench_b_tree_family,
    bench_tango_tree_family,
    bench_wavl_tree_family,
    bench_weight_balanced_tree_family,
)

_FAMILY_BENCHES = [
    bench_b_tree_family,
    bench_b_plus_tree_family,
    bench_b_star_tree_family,
    bench_weight_balanced_tree_family,
    bench_wavl_tree_family,
    bench_tango_tree_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
