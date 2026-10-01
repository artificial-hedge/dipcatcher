"""Unit tests for quant_fund.models.particle_gibbs."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.particle_gibbs import (
    _bootstrap_loglik,
    bench_particle_gibbs,
    conditional_smc,
    kalman_smoother_mean,
    particle_gibbs,
)


def _ssm(phi: float = 0.9, sd_x: float = 0.4, sd_y: float = 0.7, t_n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = np.empty(t_n)
    x[0] = sd_x / np.sqrt(1 - phi * phi) * rng.standard_normal()
    for t in range(1, t_n):
        x[t] = phi * x[t - 1] + sd_x * rng.standard_normal()
    y = x + sd_y * rng.standard_normal(t_n)
    return x, y


def test_kalman_smoother_tracks_state() -> None:
    x, y = _ssm()
    sm = kalman_smoother_mean(y, 0.9, 0.4, 0.7)
    assert np.corrcoef(sm, x)[0, 1] > 0.7
    assert sm.shape == x.shape


def test_conditional_smc_shape_and_finiteness() -> None:
    _, y = _ssm()
    ref = kalman_smoother_mean(y, 0.9, 0.4, 0.7)
    p = conditional_smc(y, 0.9, 0.4, 0.7, ref, n_particles=32, seed=0)
    assert p.shape == y.shape
    assert np.all(np.isfinite(p))


def test_bootstrap_loglik_prefers_true_phi() -> None:
    _, y = _ssm()
    ll_true = _bootstrap_loglik(y, 0.9, 0.4, 0.7, 128, seed=0)
    ll_far = _bootstrap_loglik(y, 0.2, 0.4, 0.7, 128, seed=0)
    assert ll_true > ll_far


def test_particle_gibbs_draws() -> None:
    _, y = _ssm(t_n=50)
    out = particle_gibbs(y, 0.85, 0.4, 0.7, n_iter=30, n_particles=32, seed=0)
    draws = np.asarray(out["path_draws"])
    assert draws.shape == (30, 50)
    assert np.all(np.isfinite(np.asarray(out["phi_draws"])))


def test_particle_gibbs_rejects_bad_params() -> None:
    _, y = _ssm()
    with pytest.raises(ValueError):
        particle_gibbs(y, 1.5, 0.4, 0.7, n_iter=5, seed=0)
    with pytest.raises(ValueError):
        particle_gibbs(y[:10], 0.9, 0.4, 0.7, n_iter=5, seed=0)


def test_bench_particle_gibbs_score() -> None:
    out = bench_particle_gibbs()
    assert out["score"] == pytest.approx(1.0)
    assert out["synthetic_pg_phi_err"] < 0.12
