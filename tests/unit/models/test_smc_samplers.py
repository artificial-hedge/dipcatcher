"""Unit tests for quant_fund.models.smc_samplers."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.smc_samplers import _ess, bench_smc_samplers, smc_tempered


def _gauss_target(y_obs: float = 1.0, sd_l: float = 0.5, sd_p: float = 3.0):
    def logprior(x):
        return (-0.5 * (x / sd_p) ** 2 - np.log(sd_p) - 0.5 * np.log(2 * np.pi)).sum(axis=1)

    def loglike(x):
        return (-0.5 * ((x - y_obs) / sd_l) ** 2 - np.log(sd_l) - 0.5 * np.log(2 * np.pi)).sum(
            axis=1
        )

    def sp(rng):
        return rng.standard_normal(x_d) * sd_p

    x_d = 1
    return logprior, loglike, sp


def test_ess_weights() -> None:
    uniform = np.full(100, 0.01)
    assert _ess(uniform) == pytest.approx(100.0)
    degenerate = np.zeros(100)
    degenerate[0] = 1.0
    assert _ess(degenerate) == pytest.approx(1.0)


def test_smc_returns_particles_and_logz() -> None:
    lp, ll, sp = _gauss_target()
    out = smc_tempered(lp, ll, sp, n_particles=200, n_beta=10, n_mcmc=2, seed=0)
    parts = np.asarray(out["particles"])
    assert parts.shape[0] == 200
    assert "logz_smc" in out
    assert np.isfinite(out["logz_smc"])


def test_smc_particles_move_to_posterior() -> None:
    y = 2.0
    lp, ll, sp = _gauss_target(y_obs=y)
    out = smc_tempered(lp, ll, sp, n_particles=300, n_beta=15, n_mcmc=3, seed=1)
    parts = np.asarray(out["particles"])
    # posterior mean ~ 1.8 for sd_l=0.5, sd_p=3
    assert abs(parts.mean() - y) < 0.5
    assert parts.std() < 1.0


def test_smc_deterministic() -> None:
    lp, ll, sp = _gauss_target()
    a = smc_tempered(lp, ll, sp, n_particles=100, n_beta=8, n_mcmc=2, seed=7)
    b = smc_tempered(lp, ll, sp, n_particles=100, n_beta=8, n_mcmc=2, seed=7)
    np.testing.assert_array_equal(np.asarray(a["particles"]), np.asarray(b["particles"]))


def test_smc_rejects_bad_scores() -> None:
    def lp(x):
        return np.zeros(x.shape[0])

    def bad_ll(x):
        return np.full(x.shape[0], np.nan)

    def sp(rng):
        return rng.standard_normal(1)

    with pytest.raises(ValueError):
        smc_tempered(lp, bad_ll, sp, n_particles=50, n_beta=5, seed=0)


def test_bench_smc_samplers_score() -> None:
    out = bench_smc_samplers()
    assert out["synthetic_score"] == pytest.approx(1.0)
    assert out["synthetic_smc_mean_err"] < 0.15
