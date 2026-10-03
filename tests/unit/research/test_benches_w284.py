"""Adapter tests for wave-284 bioinformatics-3 canon benches."""

from quant_fund.research.benches_w284 import (
    bench_band_align_family,
    bench_codon_usage_family,
    bench_fitch_pars_family,
    bench_jc69_lik_family,
    bench_nj_tree_family,
    bench_seed_extend_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_nj_tree_family,
        bench_fitch_pars_family,
        bench_seed_extend_family,
        bench_band_align_family,
        bench_jc69_lik_family,
        bench_codon_usage_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
