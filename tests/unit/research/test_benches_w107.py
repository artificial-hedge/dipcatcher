import numpy as np
import pytest

from quant_fund.research import benches_w107 as bw

FAMILIES = [
    bw.bench_adi_family,
    bw.bench_lax_wendroff_family,
    bw.bench_weno_family,
    bw.bench_level_set_family,
    bw.bench_fast_marching_family,
    bw.bench_godunov_family,
]


@pytest.mark.parametrize("fn", FAMILIES, ids=lambda f: f.__name__)
def test_family_runs_and_finite(fn):
    out = fn()
    assert isinstance(out, dict) and out
    for k, v in out.items():
        assert isinstance(v, float)
        assert np.isfinite(v)
        assert not any(b in k.lower() for b in ("sharpe", "sortino", "calmar", "pnl", "nav"))
