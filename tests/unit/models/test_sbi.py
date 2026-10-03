"""Tests for simulation-based inference (models/sbi.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.sbi import (
    abc_rej,
    abc_smc,
    bench_sbi,
    prior_ma2,
    ratio_estimator,
    synth_gk,
    synth_ma2,
)


@pytest.fixture
def obs():
    th = np.array([0.8, -0.3])
    x = synth_ma2(th, np.random.default_rng(42), t=300)
    return th, x


def test_ma2_summary_dims():
    x = synth_ma2(np.array([0.5, 0.2]), np.random.default_rng(0), t=200)
    assert x.shape == (3,)
    assert np.all(np.isfinite(x))


def test_ma2_validation():
    with pytest.raises(ValueError):
        synth_ma2(np.array([0.5]), np.random.default_rng(0))


def test_prior_bounds():
    th = prior_ma2(np.random.default_rng(0), 100)
    assert th.shape == (100, 2)
    assert np.all((th[:, 0] >= -2) & (th[:, 0] <= 2))
    assert np.all((th[:, 1] >= -1) & (th[:, 1] <= 1))


def test_abc_rej_hits_target(obs):
    th_true, x_obs = obs
    out = abc_rej(
        lambda t, r: synth_ma2(t, r, t=300), prior_ma2, x_obs, tol=0.12, n_accept=60, seed=1
    )
    assert out["n_accept"] == 60
    assert out["theta"].shape == (60, 2)
    err = np.linalg.norm(out["theta"].mean(axis=0) - th_true)
    assert err < 0.3


def test_abc_rej_budget_respected(obs):
    _th, x_obs = obs
    out = abc_rej(
        lambda t, r: synth_ma2(t, r, t=100),
        prior_ma2,
        x_obs,
        tol=1e-9,
        n_accept=50,
        max_trials=200,
        seed=1,
    )
    assert out["n_accept"] < 50
    assert out["trials"] <= 200


def test_abc_rej_determinism(obs):
    _th, x_obs = obs
    a = abc_rej(lambda t, r: synth_ma2(t, r, t=50), prior_ma2, x_obs, tol=0.5, n_accept=15, seed=3)
    b = abc_rej(lambda t, r: synth_ma2(t, r, t=50), prior_ma2, x_obs, tol=0.5, n_accept=15, seed=3)
    assert np.array_equal(a["theta"], b["theta"])


def test_abc_smc_tightens(obs):
    th_true, x_obs = obs
    out = abc_smc(
        lambda t, r: synth_ma2(t, r, t=300),
        prior_ma2,
        x_obs,
        n_particles=100,
        rounds=3,
        tol_schedule=np.array([0.3, 0.15, 0.07]),
        seed=4,
    )
    assert out["particles"].shape[1] == 2
    assert out["weights"].sum() == pytest.approx(1.0)
    assert out["ess"][-1] > 20
    pm = np.average(out["particles"], axis=0, weights=out["weights"])
    assert np.linalg.norm(pm - th_true) < 0.3


def test_abc_smc_schedule_validation(obs):
    _th, x_obs = obs
    with pytest.raises(ValueError):
        abc_smc(
            lambda t, r: synth_ma2(t, r, t=100),
            prior_ma2,
            x_obs,
            n_particles=50,
            rounds=2,
            tol_schedule=np.array([0.1, 0.2]),  # increasing
            seed=1,
        )


def test_ratio_estimator_ranks(obs):
    th_true, x_obs = obs
    lr = ratio_estimator(
        lambda t, r: synth_ma2(t, r, t=80),
        prior_ma2,
        theta_dim=2,
        n_train=1500,
        iters=200,
        seed=6,
    )
    assert lr(th_true, x_obs) > lr(np.array([-1.8, 0.9]), x_obs)


def test_gk_simulator():
    x = synth_gk(np.array([3.0, 1.0, 2.0, 0.5]), np.random.default_rng(0), n=200)
    assert x.shape == (8,)
    assert np.all(np.diff(x) > 0)  # quantiles increasing


def test_bench_keys():
    out = bench_sbi()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_abc_posterior_err"] < 0.4
    assert out["synthetic_smc_posterior_err"] < 0.4
    assert out["synthetic_smc_ess_final"] > 20
    assert out["synthetic_nre_margin"] > 0.0
    assert out["synthetic_determinism"] == 1.0


def test_input_validation(obs):
    _th, x_obs = obs
    with pytest.raises(ValueError):
        abc_rej(lambda t, r: synth_ma2(t, r, t=50), prior_ma2, x_obs, tol=-1.0)
    with pytest.raises(ValueError):
        abc_smc(lambda t, r: synth_ma2(t, r, t=50), prior_ma2, x_obs, n_particles=5, rounds=2)
