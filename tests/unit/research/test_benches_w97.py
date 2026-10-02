import numpy as np
import pytest

from quant_fund.research.benches_w97 import (
    bench_multiclass_family,
    bench_ode_solvers_family,
    bench_proximal_gradient_family,
    bench_quadrature_family,
    bench_semisupervised_family,
    bench_sparse_pca_family,
)
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

_FAMILIES = {
    "proximal_gradient": bench_proximal_gradient_family,
    "quadrature": bench_quadrature_family,
    "ode_solvers": bench_ode_solvers_family,
    "semisupervised": bench_semisupervised_family,
    "multiclass": bench_multiclass_family,
    "sparse_pca": bench_sparse_pca_family,
}


def test_w97_families_registered():
    for name in _FAMILIES:
        assert name in OPTIONAL_BENCHMARK_FAMILIES, name


@pytest.mark.parametrize("name,fn", list(_FAMILIES.items()))
def test_w97_bench_outputs_finite(name, fn):
    out = fn()
    assert isinstance(out, dict) and len(out) > 0
    for k, v in out.items():
        assert isinstance(k, str)
        assert isinstance(v, float)
        assert np.isfinite(v), (name, k, v)
