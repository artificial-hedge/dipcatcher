"""BG/NBD CLV tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.pareto_nbd import (
    bench_pnbd,
    bgnbd_expected_purchases,
    bgnbd_loglik,
    fit_bgnbd,
    p_alive_bgnbd,
)


def _cbs(seed=0, n=80):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(1.0, 4.0, n)
    T = np.full(n, 52.0)
    x = np.zeros(n)
    tx = np.zeros(n)
    for i in range(n):
        t = 0.0
        while t < 52.0:
            t += rng.exponential(52.0 / lam[i])
            if t < 52.0:
                x[i] += 1
                tx[i] = t
    return x, tx, T


def test_loglik_finite():
    x, tx, T = _cbs()
    ll = bgnbd_loglik(1.0, 5.0, 1.0, 3.0, x, tx, T)
    assert np.isfinite(ll) and ll < 0


def test_loglik_bad_params():
    x, tx, T = _cbs()
    assert bgnbd_loglik(-1, 5, 1, 3, x, tx, T) == -np.inf


def test_expected_purchases_nonneg():
    x, tx, T = _cbs()
    pred = bgnbd_expected_purchases(1.0, 5.0, 1.0, 3.0, x, tx, T, 10.0)
    assert (pred >= 0).all() and np.isfinite(pred).all()


def test_p_alive_bounds():
    x, tx, T = _cbs()
    pa = p_alive_bgnbd(1.0, 5.0, 1.0, 3.0, x, tx, T)
    assert ((pa >= 0) & (pa <= 1)).all()
    # zero-activity customer: P(alive) = 1 (no dropout observed)
    assert pa[x == 0].mean() > pa[x > 0].mean()


def test_fit_runs():
    x, tx, T = _cbs()
    fit = fit_bgnbd(x, tx, T)
    assert fit["r"] > 0 and np.isfinite(fit["neg_ll"])


def test_bench_pnbd():
    out = bench_pnbd()
    assert out["synthetic_mse_model"] < out["synthetic_mse_naive"] * 1.15
