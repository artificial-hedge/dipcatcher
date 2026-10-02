import numpy as np
import pytest

from quant_fund.research import benches_w103 as bw

FAMILIES = [
    bw.bench_lanczos_family,
    bw.bench_arnoldi_gmres_family,
    bw.bench_randomized_svd_family,
    bw.bench_nystrom_family,
    bw.bench_cur_decomp_family,
    bw.bench_interpolative_decomp_family,
]


@pytest.mark.parametrize("fn", FAMILIES, ids=lambda f: f.__name__)
def test_family_runs_and_finite(fn):
    out = fn()
    assert isinstance(out, dict) and out
    for k, v in out.items():
        assert isinstance(v, float)
        assert np.isfinite(v)
        assert not any(b in k.lower() for b in ("sharpe", "sortino", "calmar", "pnl", "nav"))
