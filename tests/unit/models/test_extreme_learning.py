import numpy as np

from quant_fund.models.extreme_learning import (
    bench_extreme_learning,
    elm_fit,
    elm_predict,
)


def test_elm_learns_nonlinear():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, (300, 2))
    y = np.sin(3 * x[:, 0]) + x[:, 1] ** 2
    m = elm_fit(x, y, n_hidden=200, seed=0)
    p = elm_predict(m, x)
    assert np.sqrt(np.mean((p - y) ** 2)) < 0.35


def test_elm_acts():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((50, 3))
    y = x[:, 0]
    for act in ("sigmoid", "tanh", "relu"):
        m = elm_fit(x, y, 50, act=act, seed=0)
        p = elm_predict(m, x, act=act)
        assert np.isfinite(p).all()


def test_bench_keys():
    out = bench_extreme_learning()
    assert out["synthetic_elm_vs_ridge"] > 0.2
    assert out["synthetic_elm_cls_acc"] > 0.8
