import numpy as np

from quant_fund.models.nonstationary_bandits import (
    bench_nonstationary_bandits,
    d_ucb,
    sw_ucb,
)


def test_sw_ucb_runs():
    rng = np.random.default_rng(0)
    seq = np.full((400, 3), 0.3)
    seq[:200, 0] = 0.8
    seq[200:, 1] = 0.8
    assert sw_ucb(seq, rng, window=100) >= 0.0


def test_d_ucb_runs():
    rng = np.random.default_rng(1)
    seq = np.full((400, 3), 0.3)
    seq[:200, 0] = 0.8
    seq[200:, 1] = 0.8
    assert d_ucb(seq, rng, gamma_disc=0.99) >= 0.0


def test_bench_keys():
    out = bench_nonstationary_bandits(seed=8)
    assert {"synthetic_sw_ucb_regret", "synthetic_d_ucb_regret", "synthetic_ucb1_regret"} <= set(
        out
    )
    assert all(np.isfinite(v) for v in out.values())
