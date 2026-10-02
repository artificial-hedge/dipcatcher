"""Unit tests for quant_fund.models.synthetic_likelihood."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.synthetic_likelihood import (
    bench_synthetic_likelihood,
    bsl_mcmc,
    synthetic_loglik,
)


def _problem():
    theta_true = np.array([2.0])
    sd_obs = 1.0
    n_data = 25
    rng = np.random.default_rng(0)
    data = theta_true + sd_obs * rng.standard_normal(n_data)

    def simulate(theta, r):
        return theta[0] + sd_obs * r.standard_normal(n_data)

    def summarize(dset):
        return np.asarray([dset.mean()])

    s_obs = np.asarray([data.mean()])
    return theta_true, data, simulate, summarize, s_obs


def test_synthetic_loglik_finite() -> None:
    _, _, simulate, summarize, s_obs = _problem()
    rng = np.random.default_rng(1)
    val = synthetic_loglik(np.array([2.0]), s_obs, simulate, summarize, n_sim=30, rng=rng)
    assert np.isfinite(val)


def test_synthetic_loglik_peaks_near_truth() -> None:
    _, _, simulate, summarize, s_obs = _problem()
    ll_good = synthetic_loglik(
        np.array([2.0]), s_obs, simulate, summarize, n_sim=30, rng=np.random.default_rng(2)
    )
    ll_bad = synthetic_loglik(
        np.array([5.0]), s_obs, simulate, summarize, n_sim=30, rng=np.random.default_rng(2)
    )
    assert ll_good > ll_bad


def test_bsl_mcmc_moves_to_posterior() -> None:
    theta_true, _, simulate, summarize, s_obs = _problem()
    chain = bsl_mcmc(
        np.array([0.0]),
        s_obs,
        simulate,
        summarize,
        n_iter=500,
        n_sim=15,
        step=0.3,
        burn=200,
        seed=3,
    )
    assert chain.shape == (300, 1)
    assert abs(np.median(chain[:, 0]) - theta_true[0]) < 0.8


def test_synthetic_loglik_deterministic_with_seed() -> None:
    _, _, simulate, summarize, s_obs = _problem()
    a = synthetic_loglik(
        np.array([1.5]), s_obs, simulate, summarize, n_sim=20, rng=np.random.default_rng(4)
    )
    b = synthetic_loglik(
        np.array([1.5]), s_obs, simulate, summarize, n_sim=20, rng=np.random.default_rng(4)
    )
    assert a == b


def test_bench_synthetic_likelihood_score() -> None:
    out = bench_synthetic_likelihood()
    assert out["score"] == pytest.approx(1.0)
