import numpy as np

from quant_fund.models.contextual_bandits import (
    bench_contextual_bandits,
    lin_eps_greedy,
    lin_ts,
    linucb,
)


def _env(seed=0, k=3, d=4, t=300):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((k, d)), rng.standard_normal((t, d)), rng


def test_linucb_runs():
    th, ctx, rng = _env()
    assert linucb(th, ctx, rng) >= 0.0


def test_lin_ts_runs():
    th, ctx, rng = _env()
    assert lin_ts(th, ctx, rng) >= 0.0


def test_greedy_runs():
    th, ctx, rng = _env()
    assert lin_eps_greedy(th, ctx, rng) >= 0.0


def test_bench_beats_greedy():
    out = bench_contextual_bandits(seed=5)
    assert out["synthetic_beats_greedy"] == 1.0
    assert out["synthetic_linucb_regret"] < out["synthetic_eps_greedy_regret"]
