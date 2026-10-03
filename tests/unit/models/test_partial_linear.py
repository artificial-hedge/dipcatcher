"""Tests for partially-linear regression (models/partial_linear.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.partial_linear import (
    bench_partial_linear,
    cv_bandwidth,
    kernel_smooth,
    robinson_pl,
    series_pl,
    speckman_pl,
    synth_partial_linear,
)


@pytest.fixture
def panel():
    return synth_partial_linear(n=300, corr_xz=0.6, seed=4)


def test_kernel_smooth_recovery(panel):
    z = np.asarray(panel["z"])
    g = np.asarray(panel["g_true"])
    out = kernel_smooth(g, z, bw=0.05)
    assert np.corrcoef(out, g)[0, 1] > 0.95


def test_robinson_beta(panel):
    fit = robinson_pl(np.asarray(panel["y"]), np.asarray(panel["x"]), np.asarray(panel["z"]))
    beta = np.asarray(fit["beta"])
    assert abs(beta[0] - 1.5) < 0.15
    se = np.asarray(fit["se"])
    assert se[0] > 0.0


def test_speckman_beta(panel):
    sp = speckman_pl(np.asarray(panel["y"]), np.asarray(panel["x"]), np.asarray(panel["z"]))
    assert abs(float(np.asarray(sp["beta"])[0]) - 1.5) < 0.15


def test_series_beta(panel):
    se = series_pl(
        np.asarray(panel["y"]), np.asarray(panel["x"]), np.asarray(panel["z"]), n_basis=9
    )
    assert abs(float(np.asarray(se["beta"])[0]) - 1.5) < 0.15


def test_cv_bandwidth(panel):
    cv = cv_bandwidth(np.asarray(panel["y"]), np.asarray(panel["x"]), np.asarray(panel["z"]))
    bw = float(cv["bw"])
    assert 0.001 < bw < 1.0
    assert np.asarray(cv["mse"]).size == np.asarray(cv["grid"]).size


def test_validation():
    with pytest.raises(ValueError):
        robinson_pl(np.ones(10), np.ones(10), np.ones(10))
    with pytest.raises(ValueError):
        robinson_pl(np.ones(40), np.ones(40), np.ones(30))
    with pytest.raises(ValueError):
        kernel_smooth(np.ones(30), np.ones(30), bw=0.0)
    with pytest.raises(ValueError):
        series_pl(np.ones(40), np.ones(40), np.ones(40), n_basis=0)


def test_determinism(panel):
    a = robinson_pl(np.asarray(panel["y"]), np.asarray(panel["x"]), np.asarray(panel["z"]))["beta"]
    b = robinson_pl(np.asarray(panel["y"]), np.asarray(panel["x"]), np.asarray(panel["z"]))["beta"]
    assert np.allclose(a, b)


def test_bench_keys():
    out = bench_partial_linear()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_beta_err"] < 0.2
    assert out["synthetic_semi_beats_ols"] == 1.0
    assert out["synthetic_g_corr"] > 0.9
    assert out["synthetic_determinism"] == 1.0
