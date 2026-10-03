import numpy as np
import pytest

from quant_fund.research import benches_w105 as bw

FAMILIES = [
    bw.bench_alpha_beta_family,
    bw.bench_mcts_family,
    bw.bench_puct_family,
    bw.bench_negascout_family,
    bw.bench_proof_number_family,
    bw.bench_dfpn_family,
]


@pytest.mark.parametrize("fn", FAMILIES, ids=lambda f: f.__name__)
def test_family_runs_and_finite(fn):
    out = fn()
    assert isinstance(out, dict) and out
    for k, v in out.items():
        assert isinstance(v, float)
        assert np.isfinite(v)
        assert not any(b in k.lower() for b in ("sharpe", "sortino", "calmar", "pnl", "nav"))
