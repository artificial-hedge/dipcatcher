"""Tests for Cragg two-part hurdle model (models/hurdle.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.hurdle import bench_hurdle, hurdle_fit, synth_hurdle


def test_gamma_recovered():
    d = synth_hurdle(gamma_x=0.8, seed=37)
    out = hurdle_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert abs(float(out["gamma_x"]) - 0.8) < 0.4


def test_beta_recovered():
    d = synth_hurdle(beta_x=1.5, seed=37)
    out = hurdle_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert abs(float(out["beta_x"]) - 1.5) < 0.4


def test_fit_correlation():
    d = synth_hurdle(seed=37)
    out = hurdle_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert float(out["fit_cor"]) > 0.6


def test_zero_share():
    d = synth_hurdle(seed=37)
    out = hurdle_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    expected = float(np.mean(np.asarray(d["y"]) == 0.0))
    assert abs(float(out["share_zero"]) - expected) < 1e-9


def test_validation():
    d = synth_hurdle(seed=37)
    y, x = np.asarray(d["y"]), np.asarray(d["x"])
    with pytest.raises(ValueError):
        hurdle_fit(-np.abs(y), x)  # negatives not allowed
    with pytest.raises(ValueError):
        hurdle_fit(y[:40], x[:40])  # too few rows
    with pytest.raises(ValueError):
        hurdle_fit(y, x[:, :1] * 0.0)  # constant covariate
    with pytest.raises(ValueError):
        hurdle_fit(y + 1.0, x)  # no zeros → no hurdle
    with pytest.raises(ValueError):
        hurdle_fit(np.where(y == 0, 1.0, y), x)  # all positive


def test_determinism():
    d = synth_hurdle(seed=37)
    a = hurdle_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    b = hurdle_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert float(a["gamma_x"]) == float(b["gamma_x"])
    assert float(a["beta_x"]) == float(b["beta_x"])


def test_bench_keys():
    out = bench_hurdle()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
