"""Adapter tests for wave-288 measure-theory canon benches."""

from quant_fund.research.benches_w288 import (
    bench_conv_prob_family,
    bench_fubini_swap_family,
    bench_leb_integral_family,
    bench_leb_measure_family,
    bench_radon_nikodym_family,
    bench_weak_conv_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_leb_measure_family,
        bench_leb_integral_family,
        bench_conv_prob_family,
        bench_weak_conv_family,
        bench_fubini_swap_family,
        bench_radon_nikodym_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
