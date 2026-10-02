import numpy as np

from quant_fund.models.stochastic_bandits import (
    bench_stochastic_bandits,
    epsilon_greedy,
    explore_then_commit,
    ucb1,
)


def test_ucb1_finds_best():
    rng = np.random.default_rng(0)
    probs = np.array([0.7, 0.4, 0.4])
    r = ucb1(probs, 500, rng)
    assert 0.0 <= r < 500 * 0.3


def test_eps_greedy_regret_nonnegative():
    rng = np.random.default_rng(1)
    probs = np.array([0.6, 0.5])
    assert epsilon_greedy(probs, 300, rng) >= 0.0


def test_etc_commits():
    rng = np.random.default_rng(2)
    probs = np.array([0.9, 0.1])
    r = explore_then_commit(probs, 400, rng, m=10)
    assert r < 400 * 0.8


def test_bench_keys():
    out = bench_stochastic_bandits(seed=3)
    assert {"synthetic_ucb1_regret", "synthetic_eps_greedy_regret", "synthetic_etc_regret"} <= set(
        out
    )
    assert all(np.isfinite(v) for v in out.values())
