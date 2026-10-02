import numpy as np
import pytest

from quant_fund.research import benches_w102 as bw

FAMILIES = [
    bw.bench_stochastic_bandits_family,
    bw.bench_kl_bandits_family,
    bw.bench_contextual_bandits_family,
    bw.bench_adversarial_bandits_family,
    bw.bench_best_arm_family,
    bw.bench_nonstationary_bandits_family,
]


@pytest.mark.parametrize("fn", FAMILIES, ids=lambda f: f.__name__)
def test_family_runs_and_finite(fn):
    out = fn()
    assert isinstance(out, dict) and out
    for k, v in out.items():
        assert isinstance(v, float)
        assert np.isfinite(v)
        assert not any(b in k.lower() for b in ("sharpe", "sortino", "calmar", "pnl", "nav"))
