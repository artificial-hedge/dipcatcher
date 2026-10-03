"""Tests for Heckman selection models (models/heckman.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.heckman import (
    bench_heckman,
    heckman_ml,
    heckman_two_step,
    inverse_mills,
    synth_heckman,
)


@pytest.fixture
def panel():
    return synth_heckman(n=700, rho=0.7, seed=4)


def test_inverse_mills():
    z = np.linspace(-3, 3, 50)
    lam = inverse_mills(z)
    assert np.all(lam > 0)
    assert np.all(np.diff(lam) < 0)  # decreasing
    assert abs(inverse_mills(np.array([0.0]))[0] - math.sqrt(2 / math.pi)) < 0.05


def test_two_step_beta(panel):
    ts = heckman_two_step(
        np.asarray(panel["y"]),
        np.asarray(panel["x"]),
        np.asarray(panel["w"]),
        np.asarray(panel["s"]),
    )
    assert abs(float(np.asarray(ts["beta"])[0]) - 1.0) < 0.15
    assert float(ts["rho_sigma"]) > 0.0  # detects positive selection


def test_ml_returns(panel):
    ml = heckman_ml(
        np.asarray(panel["y"]),
        np.asarray(panel["x"]),
        np.asarray(panel["w"]),
        np.asarray(panel["s"]),
    )
    assert math.isfinite(float(ml["loglik"]))
    assert -1.0 <= float(ml["rho"]) <= 1.0
    assert float(ml["sigma"]) > 0.0


def test_selection_correction_direction(panel):
    # subsample OLS slope differs from truth; Heckman corrects toward it
    sel = np.asarray(panel["s"]) == 1.0
    xo = np.column_stack([np.ones(int(sel.sum())), np.asarray(panel["x"])[sel]])
    beta_ols = np.linalg.lstsq(xo, np.asarray(panel["y"])[sel], rcond=None)[0][1]
    ts = heckman_two_step(
        np.asarray(panel["y"]),
        np.asarray(panel["x"]),
        np.asarray(panel["w"]),
        np.asarray(panel["s"]),
    )
    beta_ts = float(np.asarray(ts["beta"])[0])
    assert abs(beta_ts - 1.0) <= abs(beta_ols - 1.0) + 0.05


def test_validation():
    with pytest.raises(ValueError):
        heckman_two_step(np.ones(20), np.ones(20), np.ones(20), np.ones(20))
    with pytest.raises(ValueError):
        heckman_two_step(np.ones(60), np.ones(60), np.ones(60), np.full(60, 0.5))
    with pytest.raises(ValueError):
        heckman_two_step(np.ones(60), np.ones(60), np.ones(60), np.ones(60))
    with pytest.raises(ValueError):
        heckman_ml(np.ones(60), np.ones(60), np.ones(60), np.zeros(60))


def test_determinism(panel):
    a = heckman_two_step(
        np.asarray(panel["y"]),
        np.asarray(panel["x"]),
        np.asarray(panel["w"]),
        np.asarray(panel["s"]),
    )["beta"]
    b = heckman_two_step(
        np.asarray(panel["y"]),
        np.asarray(panel["x"]),
        np.asarray(panel["w"]),
        np.asarray(panel["s"]),
    )["beta"]
    assert np.allclose(a, b)


def test_bench_keys():
    out = bench_heckman()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_beta_twostep_err"] < 0.2
    assert out["synthetic_twostep_beats_ols"] == 1.0
    assert out["synthetic_determinism"] == 1.0
