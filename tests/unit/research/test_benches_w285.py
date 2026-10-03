"""Adapter tests for wave-285 information-theory canon benches."""

from quant_fund.research.benches_w285 import (
    bench_blahut_arimoto_family,
    bench_elias_gamma_family,
    bench_kl_knn_family,
    bench_markov_entropy_family,
    bench_miller_madow_family,
    bench_type_class_family,
)


def test_families_return_synthetic_scores():
    for fn in [
        bench_markov_entropy_family,
        bench_blahut_arimoto_family,
        bench_kl_knn_family,
        bench_type_class_family,
        bench_elias_gamma_family,
        bench_miller_madow_family,
    ]:
        out = fn()
        assert out
        assert all(k.startswith("synthetic_") for k in out)
