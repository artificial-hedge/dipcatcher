"""Wave-930 adapter test."""

from __future__ import annotations

from quant_fund.research.benches_w930 import (
    bench_grasp_meta_family,
    bench_iterated_local_family,
    bench_lin_kernighan_family,
    bench_tabu_search_family,
    bench_three_opt_move_family,
    bench_two_opt_move_family,
)

_FAMILY_BENCHES = [
    bench_lin_kernighan_family,
    bench_two_opt_move_family,
    bench_three_opt_move_family,
    bench_tabu_search_family,
    bench_iterated_local_family,
    bench_grasp_meta_family,
]


def test_families_emit_synthetic_scores():
    for fn in _FAMILY_BENCHES:
        blob = fn()
        assert len(blob) == 1
        key, val = next(iter(blob.items()))
        assert key.startswith("synthetic_")
        assert 0.0 <= val <= 1.0
