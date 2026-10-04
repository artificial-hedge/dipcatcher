"""Adapter tests for wave-307 regex-2 canon benches."""

from quant_fund.research.benches_w307 import (
    bench_bitap_fuzzy_family,
    bench_glushkov_nfa_family,
    bench_lazy_dfa_family,
    bench_literal_prefilter_family,
    bench_pike_vm_family,
    bench_regex_simplify_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_pike_vm_family,
        bench_lazy_dfa_family,
        bench_bitap_fuzzy_family,
        bench_literal_prefilter_family,
        bench_glushkov_nfa_family,
        bench_regex_simplify_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)


def test_scores_in_unit_interval():
    for fn in [
        bench_pike_vm_family,
        bench_lazy_dfa_family,
        bench_bitap_fuzzy_family,
        bench_literal_prefilter_family,
        bench_glushkov_nfa_family,
        bench_regex_simplify_family,
    ]:
        for v in fn().values():
            assert 0.0 <= v <= 1.0
