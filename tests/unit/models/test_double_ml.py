"""Tests for double/debiased ML (models/double_ml.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.double_ml import (
    bench_double_ml,
    dml_irm,
    dml_plr,
    synth_irm,
    synth_plr,
)


@pytest.fixture
def plr():
    return synth_plr(n=800, theta=0.8, seed=5)


def test_plr_recovers_theta(plr):
    out = dml_plr(
        np.asarray(plr["y"]), np.asarray(plr["d"]), np.asarray(plr["x"]), n_folds=4, learner="ridge"
    )
    assert abs(out["theta"] - 0.8) < 0.15
    assert out["t"] > 3.0


def test_plr_beats_ols(plr):
    y = np.asarray(plr["y"])
    d = np.asarray(plr["d"])
    xd = np.column_stack([np.ones(y.size), d])
    b_ols = float(np.linalg.lstsq(xd, y, rcond=None)[0][1])
    out = dml_plr(y, d, np.asarray(plr["x"]), learner="ridge")
    assert abs(out["theta"] - 0.8) < abs(b_ols - 0.8)


def test_plr_krr_path(plr):
    out = dml_plr(
        np.asarray(plr["y"]), np.asarray(plr["d"]), np.asarray(plr["x"]), n_folds=3, learner="krr"
    )
    assert abs(out["theta"] - 0.8) < 0.4


def test_irm_ate():
    d = synth_irm(n=800, tau=1.2, seed=6)
    out = dml_irm(np.asarray(d["y"]), np.asarray(d["d"]), np.asarray(d["x"]), learner="ridge")
    assert abs(out["ate"] - 1.6) < 0.35
    assert 0.0 < out["propensity_min"] <= out["propensity_max"] < 1.0


def test_zero_effect_null():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (400, 3))
    d = x[:, 0] + rng.normal(0, 1, 400)
    y = x[:, 1] + rng.normal(0, 1, 400)
    out = dml_plr(y, d, x, learner="ridge")
    assert abs(out["theta"]) < 0.2


def test_degenerate_r():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (200, 2))
    d = np.full(200, 0.5)  # constant D — no residual variation after partialling out
    y = rng.standard_normal(200)
    with pytest.raises(ValueError):
        dml_plr(y, d, x, learner="krr")


def test_validation():
    with pytest.raises(ValueError):
        dml_plr(np.ones(30), np.ones(30), np.ones((30, 2)))
    with pytest.raises(ValueError):
        dml_plr(np.ones(100), np.ones(50), np.ones((100, 2)))
    with pytest.raises(ValueError):
        dml_irm(
            np.ones(100), np.ones(100), np.ones((100, 2)), n_folds=2
        )  # not binary at all... all-1 d


def test_determinism(plr):
    a = dml_plr(np.asarray(plr["y"]), np.asarray(plr["d"]), np.asarray(plr["x"]), learner="ridge")[
        "theta"
    ]
    b = dml_plr(np.asarray(plr["y"]), np.asarray(plr["d"]), np.asarray(plr["x"]), learner="ridge")[
        "theta"
    ]
    assert a == b


def test_bench_keys():
    out = bench_double_ml()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_theta_err"] < 0.15
    assert out["synthetic_theta_err"] < out["synthetic_ols_err"]
    assert out["synthetic_ate_err"] < 0.35
    assert out["synthetic_determinism"] == 1.0
