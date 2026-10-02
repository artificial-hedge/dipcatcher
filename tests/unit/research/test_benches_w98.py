import numpy as np
import pytest

from quant_fund.research.benches_w98 import (
    bench_belief_propagation_family,
    bench_cross_entropy_method_family,
    bench_extreme_learning_family,
    bench_mdp_solvers_family,
    bench_nearest_centroid_family,
    bench_td_learning_family,
)
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

_FAMILIES = {
    "mdp_solvers": bench_mdp_solvers_family,
    "td_learning": bench_td_learning_family,
    "belief_propagation": bench_belief_propagation_family,
    "extreme_learning": bench_extreme_learning_family,
    "nearest_centroid": bench_nearest_centroid_family,
    "cross_entropy_method": bench_cross_entropy_method_family,
}


def test_w98_families_registered():
    for name in _FAMILIES:
        assert name in OPTIONAL_BENCHMARK_FAMILIES, name


@pytest.mark.parametrize("name,fn", list(_FAMILIES.items()))
def test_w98_bench_outputs_finite(name, fn):
    out = fn()
    assert isinstance(out, dict) and len(out) > 0
    for k, v in out.items():
        assert isinstance(k, str)
        assert isinstance(v, float)
        assert np.isfinite(v), (name, k, v)
