"""Tests for distance covariance (models/distance_covariance.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.distance_covariance import (
    bench_distance_covariance,
    dcor_pvalue,
    distance_covariance,
    synth_nonlinear,
)


def test_detects_quadratic():
    d = synth_nonlinear(kind="quadratic", noise=0.05, seed=37)
    out = distance_covariance(np.asarray(d["x"]), np.asarray(d["y"]))
    assert float(out["dcor"]) > 0.3
    assert float(out["pearson_abs"]) < 0.15


def test_independent_low():
    d = synth_nonlinear(kind="independent", seed=37)
    out = distance_covariance(np.asarray(d["x"]), np.asarray(d["y"]))
    assert float(out["dcor"]) < 0.15


def test_linear_dependence():
    rng = np.random.default_rng(37)
    x = rng.normal(0, 1, 300)
    y = 0.9 * x + rng.normal(0, 0.3, 300)
    out = distance_covariance(x, y)
    assert float(out["dcor"]) > 0.5


def test_perm_p_rejects():
    d = synth_nonlinear(kind="quadratic", noise=0.05, seed=37)
    p = dcor_pvalue(np.asarray(d["x"]), np.asarray(d["y"]), seed=37)
    assert p < 0.05


def test_perm_p_null_kept():
    d = synth_nonlinear(kind="independent", seed=37)
    p = dcor_pvalue(np.asarray(d["x"]), np.asarray(d["y"]), seed=37)
    assert p > 0.05


def test_validation():
    d = synth_nonlinear(kind="quadratic", seed=37)
    x, y = np.asarray(d["x"]), np.asarray(d["y"])
    with pytest.raises(ValueError):
        distance_covariance(x[:20], y[:20])
    with pytest.raises(ValueError):
        distance_covariance(x, np.full(y.size, 1.0))
    x_nan = x.copy()
    x_nan[0] = np.nan
    with pytest.raises(ValueError):
        distance_covariance(x_nan, y)
    with pytest.raises(ValueError):
        distance_covariance(x, y[:-1])


def test_determinism():
    d = synth_nonlinear(kind="quadratic", seed=37)
    a = distance_covariance(np.asarray(d["x"]), np.asarray(d["y"]))
    b = distance_covariance(np.asarray(d["x"]), np.asarray(d["y"]))
    assert float(a["dcor"]) == float(b["dcor"])


def test_bench_keys():
    out = bench_distance_covariance()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
