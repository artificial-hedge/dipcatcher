"""Tests for Markov-switching regime models (models/markov_switching.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.markov_switching import (
    bench_markov_switching,
    hamilton_filter,
    ms_em_fit,
    regime_forecast,
    synth_markov,
)


@pytest.fixture
def fitted():
    d = synth_markov(n=600, seed=8)
    fit = ms_em_fit(np.asarray(d["y"]), n_states=2, n_iter=20, seed=8)
    return d, fit


def test_em_recovers_means(fitted):
    d, fit = fitted
    mu_hat = np.asarray(fit["mu"])
    mu_true = np.asarray(d["mu"])
    assert np.abs(np.sort(mu_hat) - np.sort(mu_true)).max() < 0.2


def test_filtered_probs_valid(fitted):
    d, fit = fitted
    fl = np.asarray(fit["filtered"])
    assert np.allclose(fl.sum(axis=1), 1.0, atol=1e-8)
    assert ((fl >= 0) & (fl <= 1)).all()


def test_smoother_tracks_states(fitted):
    d, fit = fitted
    sm = np.asarray(fit["smoothed"])
    pred_hi = (sm[:, 1] > 0.5).astype(np.float64)
    agree = float(np.mean(pred_hi == np.asarray(d["states"])))
    assert agree > 0.85


def test_filter_matches_em_loglik(fitted):
    d, fit = fitted
    f = hamilton_filter(
        np.asarray(d["y"]), np.asarray(fit["mu"]), np.asarray(fit["sigma"]), np.asarray(fit["p"])
    )
    assert abs(float(f["loglik"]) - float(fit["loglik"])) < 1e-6


def test_persistence(fitted):
    _, fit = fitted
    p = np.asarray(fit["p"])
    persist = float(p[0, 0] + p[1, 1]) / 2
    assert persist > 0.8


def test_forecast(fitted):
    d, fit = fitted
    f = hamilton_filter(
        np.asarray(d["y"]), np.asarray(fit["mu"]), np.asarray(fit["sigma"]), np.asarray(fit["p"])
    )
    fc = regime_forecast(f, h=10)
    assert abs(np.asarray(fc["xi_h"]).sum() - 1.0) < 1e-6
    assert np.asarray(fc["var"])[0] > 0


def test_validation():
    with pytest.raises(ValueError):
        hamilton_filter(np.ones(30), np.array([0.0, 1.0]), np.ones(2), np.eye(2))
    with pytest.raises(ValueError):
        hamilton_filter(np.ones(100), np.array([0.0, 1.0]), np.array([-1.0, 1.0]), np.eye(2))
    with pytest.raises(ValueError):
        hamilton_filter(np.ones(100), np.array([0.0, 1.0]), np.ones(2), np.ones((2, 2)))
    with pytest.raises(ValueError):
        regime_forecast(
            {
                "filtered": np.ones((10, 2)) / 2,
                "p": np.eye(2),
                "mu": np.zeros(2),
                "sigma": np.ones(2),
            },
            h=0,
        )


def test_determinism():
    d = synth_markov(n=400, seed=3)
    a = ms_em_fit(np.asarray(d["y"]), n_states=2, n_iter=10, seed=3)["mu"]
    b = ms_em_fit(np.asarray(d["y"]), n_states=2, n_iter=10, seed=3)["mu"]
    assert np.allclose(np.asarray(a), np.asarray(b))


def test_bench_keys():
    out = bench_markov_switching()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_mu_err"] < 0.15
    assert out["synthetic_regime_accuracy"] > 0.85
    assert out["synthetic_determinism"] == 1.0
