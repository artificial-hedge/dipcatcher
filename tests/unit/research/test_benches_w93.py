import numpy as np

from quant_fund.research.benches_w93 import (
    bench_factorization_machine,
    bench_gp_classification,
    bench_metric_learning,
    bench_one_class_classification,
    bench_phase_retrieval,
    bench_tree_ensembles,
)
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

WAVE93_FAMILIES = (
    "tree_ensembles",
    "metric_learning",
    "one_class_classification",
    "gp_classification",
    "phase_retrieval",
    "factorization_machine",
)


def test_wave93_families_in_registry():
    for fam in WAVE93_FAMILIES:
        assert fam in OPTIONAL_BENCHMARK_FAMILIES, fam


def test_wave93_adapters_emit_finite_floats():
    for bench in (
        bench_tree_ensembles,
        bench_metric_learning,
        bench_one_class_classification,
        bench_gp_classification,
        bench_phase_retrieval,
        bench_factorization_machine,
    ):
        out = bench()
        assert out, bench.__name__
        assert all(isinstance(v, float) and np.isfinite(v) for v in out.values())
