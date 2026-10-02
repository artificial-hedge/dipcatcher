import numpy as np

from quant_fund.research.benches_w92 import (
    bench_bayesian_optimization,
    bench_fuzzy_clustering,
    bench_hyperband_search,
    bench_metaheuristic_optimizers,
    bench_pagerank_topology,
    bench_self_organizing_maps,
)
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

WAVE92_FAMILIES = (
    "metaheuristic_optimizers",
    "bayesian_optimization",
    "fuzzy_clustering",
    "self_organizing_maps",
    "pagerank_topology",
    "hyperband_search",
)


def test_wave92_families_in_registry():
    for fam in WAVE92_FAMILIES:
        assert fam in OPTIONAL_BENCHMARK_FAMILIES, fam


def test_wave92_adapters_emit_finite_floats():
    for bench in (
        bench_metaheuristic_optimizers,
        bench_bayesian_optimization,
        bench_fuzzy_clustering,
        bench_self_organizing_maps,
        bench_pagerank_topology,
        bench_hyperband_search,
    ):
        out = bench()
        assert out, bench.__name__
        assert all(isinstance(v, float) and np.isfinite(v) for v in out.values())


def test_wave92_no_forbidden_tokens():
    for bench in (
        bench_metaheuristic_optimizers,
        bench_bayesian_optimization,
        bench_fuzzy_clustering,
        bench_self_organizing_maps,
        bench_pagerank_topology,
        bench_hyperband_search,
    ):
        for k in bench():
            assert not any(tok in k for tok in ("sharpe", "sortino", "calmar", "pnl", "nav")), k
