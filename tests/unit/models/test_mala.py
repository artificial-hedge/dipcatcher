"""Unit tests for quant_fund.models.mala."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.mala import bench_mala, mala, rwmh, ula


def _target():
    mu = np.array([1.0, -0.5])
    cov = np.array([[1.0, 0.7], [0.7, 1.0]])
    prec = np.linalg.inv(cov)

    def logp(x):
        d = x - mu
        return float(-0.5 * d @ prec @ d)

    def grad(x):
        return np.asarray(-(prec @ (x - mu)), dtype=np.float64)

    return mu, cov, logp, grad


def test_mala_returns_chain_and_acceptance() -> None:
    mu, _, logp, grad = _target()
    draws, acc = mala(logp, grad, np.zeros(2), n_iter=500, step=0.5, burn=100, seed=0)
    assert draws.shape == (400, 2)
    assert 0.0 < acc <= 1.0


def test_mala_recovers_gaussian_moments() -> None:
    mu, cov, logp, grad = _target()
    draws, _ = mala(logp, grad, np.zeros(2), n_iter=6000, step=0.6, burn=1500, seed=1)
    assert np.linalg.norm(draws.mean(axis=0) - mu) < 0.25
    assert np.linalg.norm(np.cov(draws.T) - cov) < 0.5


def test_mala_beats_rwm_acceptance() -> None:
    _, _, logp, grad = _target()
    _, acc_m = mala(logp, grad, np.zeros(2), n_iter=2000, step=0.6, burn=500, seed=2)
    _, acc_r = rwmh(logp, np.zeros(2), n_iter=2000, step=0.6, burn=500, seed=2)
    assert acc_m > acc_r


def test_ula_output_shape_finite() -> None:
    mu, _, logp, grad = _target()
    du = ula(logp, grad, mu.copy(), n_iter=2000, step=0.2, burn=500, seed=3)
    assert du.shape == (1500, 2)
    assert np.all(np.isfinite(du))


def test_mala_deterministic() -> None:
    _, _, logp, grad = _target()
    a, _ = mala(logp, grad, np.zeros(2), n_iter=300, step=0.5, burn=50, seed=9)
    b, _ = mala(logp, grad, np.zeros(2), n_iter=300, step=0.5, burn=50, seed=9)
    np.testing.assert_array_equal(a, b)


def test_mala_rejects_nonfinite_density() -> None:
    def bad(x):
        return float("nan")

    def grad(x):
        return np.zeros_like(x)

    with pytest.raises(ValueError):
        mala(bad, grad, np.zeros(2), n_iter=10, step=0.5, seed=0)


def test_bench_mala_score() -> None:
    out = bench_mala()
    assert out["score"] == pytest.approx(1.0)
    assert out["synthetic_mala_acc"] > out["synthetic_mala_rwm_acc"]
