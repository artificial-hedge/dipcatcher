"""Adapter tests for wave-302 compiler-5/JIT canon benches."""

from quant_fund.research.benches_w302 import (
    bench_card_table_gc_family,
    bench_escape_analysis_family,
    bench_gvn_pre_family,
    bench_osr_deopt_family,
    bench_ssa_repair_family,
    bench_trace_tree_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_card_table_gc_family,
        bench_escape_analysis_family,
        bench_osr_deopt_family,
        bench_trace_tree_family,
        bench_ssa_repair_family,
        bench_gvn_pre_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
