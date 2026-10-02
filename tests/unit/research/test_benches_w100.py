import numpy as np
import pytest

from quant_fund.research.benches_w100 import (
    bench_anderson_accel_family,
    bench_homotopy_continuation_family,
    bench_iterative_ls_family,
    bench_qmc_sequences_family,
    bench_sequence_accel_family,
    bench_symplectic_ode_family,
)
from quant_fund.research.catalog.registry import OPTIONAL_BENCHMARK_FAMILIES

_FAMILIES = {
    "homotopy_continuation": bench_homotopy_continuation_family,
    "anderson_accel": bench_anderson_accel_family,
    "sequence_accel": bench_sequence_accel_family,
    "iterative_ls": bench_iterative_ls_family,
    "qmc_sequences": bench_qmc_sequences_family,
    "symplectic_ode": bench_symplectic_ode_family,
}


def test_w100_families_registered():
    for name in _FAMILIES:
        assert name in OPTIONAL_BENCHMARK_FAMILIES, name


@pytest.mark.parametrize("name,fn", list(_FAMILIES.items()))
def test_w100_bench_outputs_finite(name, fn):
    out = fn()
    assert isinstance(out, dict) and len(out) > 0
    for k, v in out.items():
        assert isinstance(k, str)
        assert isinstance(v, float)
        assert np.isfinite(v), (name, k, v)
