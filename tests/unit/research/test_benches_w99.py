import numpy as np
import pytest

from quant_fund.research.benches_w99 import (
    bench_evidential_family,
    bench_gibbs_sampler_family,
    bench_kde_family,
    bench_laplace_approx_family,
    bench_multi_task_family,
    bench_tensor_power_family,
)
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

_FAMILIES = {
    "gibbs_sampler": bench_gibbs_sampler_family,
    "laplace_approx": bench_laplace_approx_family,
    "kde": bench_kde_family,
    "tensor_power": bench_tensor_power_family,
    "evidential": bench_evidential_family,
    "multi_task": bench_multi_task_family,
}


def test_w99_families_registered():
    for name in _FAMILIES:
        assert name in OPTIONAL_BENCHMARK_FAMILIES, name


@pytest.mark.parametrize("name,fn", list(_FAMILIES.items()))
def test_w99_bench_outputs_finite(name, fn):
    out = fn()
    assert isinstance(out, dict) and len(out) > 0
    for k, v in out.items():
        assert isinstance(k, str)
        assert isinstance(v, float)
        assert np.isfinite(v), (name, k, v)
