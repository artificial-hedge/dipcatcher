import numpy as np
import pytest

from quant_fund.research import benches_w106 as bw

FAMILIES = [
    bw.bench_bdf_family,
    bw.bench_adams_family,
    bw.bench_radau_family,
    bw.bench_strang_family,
    bw.bench_etdrk4_family,
    bw.bench_crank_nicolson_family,
]


@pytest.mark.parametrize("fn", FAMILIES, ids=lambda f: f.__name__)
def test_family_runs_and_finite(fn):
    out = fn()
    assert isinstance(out, dict) and out
    for k, v in out.items():
        assert isinstance(v, float)
        assert np.isfinite(v)
        assert not any(b in k.lower() for b in ("sharpe", "sortino", "calmar", "pnl", "nav"))
