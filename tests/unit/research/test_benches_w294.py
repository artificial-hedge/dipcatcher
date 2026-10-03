"""Adapter tests for wave-294 compiler-4 canon benches."""

from quant_fund.research.benches_w294 import (
    bench_bb_reorder_family,
    bench_cfg_simplify_family,
    bench_jump_thread_family,
    bench_modulo_sched_family,
    bench_tail_dup_family,
    bench_tree_cover_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_tree_cover_family,
        bench_modulo_sched_family,
        bench_jump_thread_family,
        bench_tail_dup_family,
        bench_cfg_simplify_family,
        bench_bb_reorder_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
