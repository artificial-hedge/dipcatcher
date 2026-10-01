"""Tests for Zellner SUR system estimation (models/sur_model.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.sur_model import bench_sur_model, sur_fit, synth_sur


def test_betas_recovered():
    d = synth_sur(beta1=0.8, beta2=-0.5, seed=37)
    out = sur_fit(d["y"], d["x"])
    assert abs(float(out["beta10"]) - 0.8) < 0.15
    assert abs(float(out["beta20"]) + 0.5) < 0.15


def test_residual_correlation():
    d = synth_sur(rho=0.7, seed=37)
    out = sur_fit(d["y"], d["x"])
    assert abs(float(out["resid_corr"]) - 0.7) < 0.2


def test_uncorrelated_system():
    d = synth_sur(rho=0.0, seed=37)
    out = sur_fit(d["y"], d["x"])
    assert abs(float(out["resid_corr"])) < 0.2


def test_sur_close_to_ols_coeffs():
    d = synth_sur(seed=37)
    out = sur_fit(d["y"], d["x"])
    assert abs(float(out["beta10"]) - float(out["beta10_ols"])) < 0.1


def test_validation():
    d = synth_sur(seed=37)
    y, x = d["y"], d["x"]
    with pytest.raises(ValueError):
        sur_fit([y[0]], [x[0]])  # single equation
    with pytest.raises(ValueError):
        sur_fit([y[0][:20], y[1][:20]], [x[0][:20], x[1][:20]])
    with pytest.raises(ValueError):
        sur_fit(y, [x[0] * 0.0, x[1]])  # constant regressor
    bad = [y[0].copy(), y[1]]
    bad[0][5] = np.nan
    with pytest.raises(ValueError):
        sur_fit(bad, x)


def test_determinism():
    d = synth_sur(seed=37)
    a = sur_fit(d["y"], d["x"])
    b = sur_fit(d["y"], d["x"])
    assert float(a["beta10"]) == float(b["beta10"])
    assert float(a["resid_corr"]) == float(b["resid_corr"])


def test_bench_keys():
    out = bench_sur_model()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
