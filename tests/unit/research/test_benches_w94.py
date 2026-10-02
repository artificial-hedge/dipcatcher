import numpy as np

from quant_fund.research.benches_w94 import (
    bench_coordinate_descent_enet_family,
    bench_discriminant_analysis,
    bench_kernel_methods_family,
    bench_lda_topics_family,
    bench_online_convex_family,
    bench_svm_classifiers,
)
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

WAVE94_FAMILIES = (
    "svm_classifiers",
    "discriminant_analysis",
    "coordinate_descent_enet",
    "kernel_methods",
    "lda_topics",
    "online_convex",
)


def test_wave94_families_in_registry():
    for fam in WAVE94_FAMILIES:
        assert fam in OPTIONAL_BENCHMARK_FAMILIES, fam


def test_wave94_adapters_emit_finite_floats():
    for bench in (
        bench_svm_classifiers,
        bench_discriminant_analysis,
        bench_coordinate_descent_enet_family,
        bench_kernel_methods_family,
        bench_lda_topics_family,
        bench_online_convex_family,
    ):
        out = bench()
        assert out, bench.__name__
        assert all(isinstance(v, float) and np.isfinite(v) for v in out.values())
