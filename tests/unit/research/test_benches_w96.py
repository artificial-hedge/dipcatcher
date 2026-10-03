import numpy as np

from quant_fund.research.benches_w96 import (
    bench_adaboost_family,
    bench_association_rules_family,
    bench_collaborative_filtering_family,
    bench_expectation_propagation_family,
    bench_naive_bayes_family,
    bench_nash_equilibrium_family,
)
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

WAVE96_FAMILIES = (
    "naive_bayes",
    "adaboost",
    "expectation_propagation",
    "collaborative_filtering",
    "association_rules",
    "nash_equilibrium",
)


def test_wave96_families_in_registry():
    for fam in WAVE96_FAMILIES:
        assert fam in OPTIONAL_BENCHMARK_FAMILIES, fam


def test_wave96_adapters_emit_finite_floats():
    for bench in (
        bench_naive_bayes_family,
        bench_adaboost_family,
        bench_expectation_propagation_family,
        bench_collaborative_filtering_family,
        bench_association_rules_family,
        bench_nash_equilibrium_family,
    ):
        out = bench()
        assert out, bench.__name__
        assert all(isinstance(v, float) and np.isfinite(v) for v in out.values())
