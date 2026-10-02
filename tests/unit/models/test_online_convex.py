import numpy as np

from quant_fund.models.online_convex import (
    bench_online_convex,
    ftrl_proximal,
    ogd_logistic,
    passive_aggressive,
    perceptron_online,
)


def _fixture(seed: int = 0, n: int = 150):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 3))
    y = (x[:, 0] + x[:, 1] + rng.normal(0, 0.3, n) > 0).astype(float)
    return x, y


def test_perceptron_separable():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (80, 2))
    y = (x[:, 0] > 0).astype(float)
    w = perceptron_online(x, y, it=10, seed=0)
    xa = np.c_[x, np.ones(len(x))]
    assert ((xa @ w > 0).astype(float) == y).mean() > 0.95


def test_pa_beats_chance():
    x, y = _fixture()
    w = passive_aggressive(x, y, it=3, seed=1)
    xa = np.c_[x, np.ones(len(x))]
    assert ((xa @ w > 0).astype(float) == y).mean() > 0.8


def test_ogd_ftrl_positive_acc():
    x, y = _fixture()
    xa = np.c_[x, np.ones(len(x))]
    w1 = ogd_logistic(x, y, it=2, seed=2)
    w2 = ftrl_proximal(x, y, seed=2)
    assert ((xa @ w1 > 0).astype(float) == y).mean() > 0.8
    assert ((xa @ w2 > 0).astype(float) == y).mean() > 0.75


def test_bench_online_convex():
    out = bench_online_convex(seed=557)
    for k in ("perceptron", "pa", "ogd", "ftrl"):
        assert out[f"synthetic_{k}_acc"] > 0.75
