import numpy as np
import pytest

from quant_fund.research import benches_w104 as bw

FAMILIES = [
    bw.bench_sinkhorn_family,
    bw.bench_emd_lp_family,
    bw.bench_gromov_wasserstein_family,
    bw.bench_unbalanced_ot_family,
    bw.bench_wasserstein_barycenter_family,
    bw.bench_fused_gromov_family,
]


@pytest.mark.parametrize("fn", FAMILIES, ids=lambda f: f.__name__)
def test_family_runs_and_finite(fn):
    out = fn()
    assert isinstance(out, dict) and out
    for k, v in out.items():
        assert isinstance(v, float)
        assert np.isfinite(v)
        assert not any(b in k.lower() for b in ("sharpe", "sortino", "calmar", "pnl", "nav"))
