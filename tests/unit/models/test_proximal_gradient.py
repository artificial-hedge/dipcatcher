import numpy as np

from quant_fund.models.proximal_gradient import (
    bench_proximal_gradient,
    fista_lasso,
    ista_lasso,
    prox_lasso_loss,
    soft_threshold,
)


def test_soft_threshold():
    x = np.array([-2.0, 0.5, -0.2, 3.0])
    out = soft_threshold(x, 1.0)
    assert np.allclose(out, [-1.0, 0.0, 0.0, 2.0])


def test_fista_recovers_sparse():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 30))
    w = np.zeros(30)
    w[[1, 5, 9]] = [2.0, -1.0, 1.0]
    y = x @ w + 0.2 * rng.standard_normal(200)
    hat = fista_lasso(x, y, 0.03, it=300)
    assert set(np.flatnonzero(np.abs(hat) > 0.1)) == {1, 5, 9}


def test_fista_beats_ista_rate():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((120, 40))
    y = x[:, 0] * 2 + 0.3 * rng.standard_normal(120)
    fi = prox_lasso_loss(x, y, ista_lasso(x, y, 0.05, it=15), 0.05)
    ff = prox_lasso_loss(x, y, fista_lasso(x, y, 0.05, it=15), 0.05)
    assert ff <= fi + 1e-9


def test_bench_keys():
    out = bench_proximal_gradient()
    assert out["synthetic_fista_support_hit"] >= 4
    assert out["synthetic_fista_vs_ridge_err"] < 1.0
