"""Tests for Gaussian-process regression (models/gaussian_process.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.gaussian_process import (
    GPModel,
    bench_gaussian_process,
    gp_fit,
    gp_predict,
    loo_cv,
    synth_gp,
)


@pytest.fixture
def fit():
    d = synth_gp(n=120, noise=0.15, seed=5)
    out = gp_fit(np.asarray(d["x"]), np.asarray(d["y"]), n_restarts=2, seed=5)
    return d, out


def test_recovers_signal(fit):
    d, out = fit
    model = out["model"]
    assert isinstance(model, GPModel)
    pred = gp_predict(model, np.asarray(d["x"]), latent=True)
    rmse = float(np.sqrt(np.mean((np.asarray(pred["mean"]) - np.asarray(d["f"])) ** 2)))
    assert rmse < 0.15


def test_noise_estimate(fit):
    _, out = fit
    assert abs(math.sqrt(out["sigma_n2"]) - 0.15) < 0.08


def test_predictive_var_shape(fit):
    d, out = fit
    model = out["model"]
    xg = np.linspace(-2, 2, 50)[:, None]
    pred = gp_predict(model, xg)
    assert np.asarray(pred["mean"]).shape == (50,)
    assert (np.asarray(pred["sd"]) > 0).all()


def test_latent_var_smaller(fit):
    d, out = fit
    model = out["model"]
    xg = np.linspace(-2, 2, 30)[:, None]
    v_lat = float(np.mean(np.asarray(gp_predict(model, xg, latent=True)["var"])))
    v_obs = float(np.mean(np.asarray(gp_predict(model, xg, latent=False)["var"])))
    assert v_lat < v_obs


def test_loo_cv(fit):
    _, out = fit
    model = out["model"]
    loo = loo_cv(model)
    assert 0 < loo["loo_rmse"] < 0.5
    assert 0.5 <= loo["loo_coverage_95"] <= 1.0


def test_uncertainty_grows_away(fit):
    _, out = fit
    model = out["model"]
    xg = np.array([[-2.0], [0.0], [10.0]])  # extrapolate far
    pred = gp_predict(model, xg, latent=True)
    sd = np.asarray(pred["sd"])
    assert sd[2] > sd[0]  # far from data → wider posterior


def test_validation():
    with pytest.raises(ValueError):
        gp_fit(np.ones(5), np.ones(5))
    with pytest.raises(ValueError):
        gp_fit(np.ones((20, 2)), np.ones(10))
    d = synth_gp(n=60, seed=1)
    fit = gp_fit(np.asarray(d["x"]), np.asarray(d["y"]), n_restarts=1)
    model = fit["model"]
    with pytest.raises(ValueError):
        gp_predict(model, np.ones((5, 3)))


def test_determinism(fit):
    d, out = fit
    fit2 = gp_fit(np.asarray(d["x"]), np.asarray(d["y"]), n_restarts=1, seed=5)
    m1, m2 = out["model"], fit2["model"]
    assert isinstance(m1, GPModel) and isinstance(m2, GPModel)
    assert np.allclose(m1.alpha[:20], m2.alpha[:20])


def test_bench_keys():
    out = bench_gaussian_process()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_grid_rmse"] < 0.15
    assert out["synthetic_sigma_n_err"] < 0.08
    assert out["synthetic_determinism"] == 1.0
