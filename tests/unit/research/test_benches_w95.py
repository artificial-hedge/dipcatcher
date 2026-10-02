import numpy as np

from quant_fund.research.benches_w95 import (
    bench_bayesian_linear_family,
    bench_conjugate_gradient_family,
    bench_evolution_strategies_family,
    bench_frank_wolfe_family,
    bench_graphical_models_family,
    bench_sparse_coding_family,
)
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

WAVE95_FAMILIES = (
    "bayesian_linear",
    "graphical_models",
    "conjugate_gradient",
    "frank_wolfe",
    "sparse_coding",
    "evolution_strategies",
)


def test_wave95_families_in_registry():
    for fam in WAVE95_FAMILIES:
        assert fam in OPTIONAL_BENCHMARK_FAMILIES, fam


def test_wave95_adapters_emit_finite_floats():
    for bench in (
        bench_bayesian_linear_family,
        bench_graphical_models_family,
        bench_conjugate_gradient_family,
        bench_frank_wolfe_family,
        bench_sparse_coding_family,
        bench_evolution_strategies_family,
    ):
        out = bench()
        assert out, bench.__name__
        assert all(isinstance(v, float) and np.isfinite(v) for v in out.values())
