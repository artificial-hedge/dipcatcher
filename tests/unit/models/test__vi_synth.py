"""Adversarial probes for _vi_synth (SYNTHETIC)."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._vi_synth import logpost, mcmc_oracle, vi_data
from quant_fund.models._vi_synth import test_logloss as _vi_logloss


def test_vi_data_deterministic():
    a = vi_data(0, n=32)
    b = vi_data(0, n=32)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_vi_data_shapes():
    X, y, Xt, yt, w = vi_data(1, n=40, d=6)
    assert X.shape == (40, 6) and Xt.shape == (40, 6)
    assert len(y) == len(yt) == 40 and w.shape == (6,)
    assert set(np.unique(y)) <= {0, 1}


@pytest.mark.parametrize("n,d", [(0, 4), (4, 0)])
def test_vi_data_hostile(n, d):
    with pytest.raises(ValueError):
        vi_data(0, n=n, d=d)


def test_logpost_finite_and_prior_pulls_zero():
    X, y, _, _, _ = vi_data(0, n=64)
    lp0 = logpost(np.zeros(5), X, y)
    lp_big = logpost(np.full(5, 40.0), X, y)
    assert np.isfinite(lp0)
    assert lp0 > lp_big  # prior penalizes huge weights


def test_logpost_hostile():
    X, y, _, _, _ = vi_data(0, n=32)
    with pytest.raises(ValueError):
        logpost(np.zeros(5), X[:0], y[:0])
    with pytest.raises(ValueError):
        logpost(np.zeros(3), X, y)  # dim mismatch
    with pytest.raises(ValueError):
        logpost(np.zeros(5), X, y[:10])  # label/row mismatch
    with pytest.raises(ValueError):
        logpost(np.zeros(5), X, y + 2)  # non-binary labels
    with pytest.raises(ValueError):
        logpost(np.zeros(5), X, y, s0=0.0)
    with pytest.raises(ValueError):
        logpost(np.zeros(5), X, y, s0=-1.0)


def test_mcmc_oracle_deterministic_and_moves():
    X, y, _, _, w_true = vi_data(2, n=80)
    m1, s1 = mcmc_oracle(X, y, seed=5, iters=400)
    m2, s2 = mcmc_oracle(X, y, seed=5, iters=400)
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(s1, s2)
    assert m1.shape == (5,) and (s1 > 0).all()
    assert np.linalg.norm(m1) < np.linalg.norm(w_true) * 3  # sane posterior


def test_mcmc_oracle_recovers_posterior_direction():
    X, y, _, _, w_true = vi_data(3, n=200)
    m, _ = mcmc_oracle(X, y, seed=1, iters=2000)
    cos = float(m @ w_true / (np.linalg.norm(m) * np.linalg.norm(w_true)))
    assert cos > 0.8


@pytest.mark.parametrize("iters", [0, 1])
def test_mcmc_oracle_vacuous(iters):
    X, y, _, _, _ = vi_data(0, n=16)
    with pytest.raises(ValueError):
        mcmc_oracle(X, y, seed=0, iters=iters)


def test_mcmc_oracle_hostile():
    X, y, _, _, _ = vi_data(0, n=16)
    with pytest.raises(ValueError):
        mcmc_oracle(X[:0], y[:0], seed=0)
    with pytest.raises(ValueError):
        mcmc_oracle(X, y[:5], seed=0)


def test_logloss_better_for_true_weights():
    X, y, Xt, yt, w_true = vi_data(4, n=128)
    good = _vi_logloss(w_true, Xt, yt)
    junk = _vi_logloss(-w_true, Xt, yt)
    assert 0 < good < junk


def test_logloss_hostile():
    _, _, Xt, yt, w = vi_data(0, n=32)
    with pytest.raises(ValueError):
        _vi_logloss(w, Xt[:0], yt[:0])
    with pytest.raises(ValueError):
        _vi_logloss(w[:3], Xt, yt)
    with pytest.raises(ValueError):
        _vi_logloss(w, Xt, yt + 5)
