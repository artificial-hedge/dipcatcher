import numpy as np
import pytest

from quant_fund.research import benches_w108 as bw

FAMILIES = [
    bw.bench_expm_pade_family,
    bw.bench_matrix_sqrt_family,
    bw.bench_sylvester_family,
    bw.bench_riccati_care_family,
    bw.bench_matrix_sign_family,
    bw.bench_toeplitz_solve_family,
]


@pytest.mark.parametrize("fn", FAMILIES, ids=lambda f: f.__name__)
def test_family_runs_and_finite(fn):
    out = fn()
    assert isinstance(out, dict) and out
    for k, v in out.items():
        assert isinstance(v, float)
        assert np.isfinite(v)
        assert not any(b in k.lower() for b in ("sharpe", "sortino", "calmar", "pnl", "nav"))
