"""Adapter tests for wave-305 text-index-2/stringology canon benches."""

from quant_fund.research.benches_w305 import (
    bench_booth_rotation_family,
    bench_lyndon_factor_family,
    bench_palindromic_tree_family,
    bench_suffix_array_lcp_family,
    bench_suffix_tree_lex_family,
    bench_z_function_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_suffix_array_lcp_family,
        bench_z_function_family,
        bench_suffix_tree_lex_family,
        bench_booth_rotation_family,
        bench_lyndon_factor_family,
        bench_palindromic_tree_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
