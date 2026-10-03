"""Tests for count-data regression (models/count_data.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.count_data import (
    bench_count_data,
    hurdle_fit,
    negbin_fit,
    poisson_fit,
    synth_negbin,
    synth_zip,
    vuong_zip_vs_poisson,
    zip_fit,
)


@pytest.fixture
def nb_panel():
    return synth_negbin(n=300, alpha=0.8, seed=7)


@pytest.fixture
def zip_panel():
    return synth_zip(n=400, pi=0.35, seed=8)


def test_poisson_fit_recovers(nb_panel):
    fit = poisson_fit(nb_panel["y"], nb_panel["x"])
    beta = np.asarray(fit["beta"])
    assert abs(beta[1] - 0.8) < 0.25
    assert fit["se"].shape == beta.shape
    assert float(fit["pearson_chi2"]) > 300.0  # overdispersed


def test_negbin_recovers_alpha(nb_panel):
    fit = negbin_fit(nb_panel["y"], nb_panel["x"])
    assert abs(float(fit["alpha"]) - 0.8) < 0.3
    assert float(fit["p_alpha"]) < 0.01
    beta = np.asarray(fit["beta"])
    assert np.linalg.norm(beta - np.array([0.5, 0.8])) < 0.15


def test_zip_fit_recovers(zip_panel):
    fit = zip_fit(zip_panel["y"], zip_panel["x"])
    pi = float(np.asarray(fit["pi_hat"]).mean())
    assert 0.2 < pi < 0.55
    beta = np.asarray(fit["beta"])
    assert np.linalg.norm(beta - np.array([0.4, 0.6])) < 0.15


def test_hurdle_fit(zip_panel):
    out = hurdle_fit(zip_panel["y"], zip_panel["x"])
    assert 0.4 < float(out["share_pos"]) < 0.9
    beta = np.asarray(out["beta_count"])
    assert np.linalg.norm(beta - np.array([0.4, 0.6])) < 0.3


def test_vuong_picks_zip(zip_panel):
    v = vuong_zip_vs_poisson(zip_panel["y"], zip_panel["x"])
    assert v["vuong_z"] > 1.96


def test_vuong_poisson_null():
    # plain Poisson data: Vuong should not favor ZIP strongly
    rng = np.random.default_rng(0)
    x = rng.standard_normal(400)
    y = rng.poisson(np.exp(0.5 + 0.4 * x)).astype(float)
    v = vuong_zip_vs_poisson(y, x)
    assert v["vuong_z"] < 5.0


def test_count_validation():
    with pytest.raises(ValueError):
        poisson_fit(np.array([0.5, 1.5] * 30), np.zeros(60))
    with pytest.raises(ValueError):
        negbin_fit(-np.ones(30), np.zeros(30))
    with pytest.raises(ValueError):
        zip_fit(np.zeros(50), np.zeros(40))
    with pytest.raises(ValueError):
        hurdle_fit(np.zeros(50), np.zeros(50))  # no positives


def test_determinism(nb_panel):
    a = negbin_fit(nb_panel["y"], nb_panel["x"])["alpha"]
    b = negbin_fit(nb_panel["y"], nb_panel["x"])["alpha"]
    assert a == b


def test_bench_keys():
    out = bench_count_data()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_nb_beta_err"] < 0.15
    assert out["synthetic_alpha_err"] < 0.3
    assert out["synthetic_lr_alpha_p"] < 0.01
    assert out["synthetic_pearson_overdisp"] > 1.5
    assert out["synthetic_vuong_picks_zip"] == 1.0
    assert out["synthetic_determinism"] == 1.0
