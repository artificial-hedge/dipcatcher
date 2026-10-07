"""Tests for p_spline."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.p_spline import bench_p_spline, p_spline


def test_smooth_recovery():
    rng = np.random.default_rng(0)
    x = np.sort(rng.random(200) * 6)
    f = np.sin(x)
    y = f + 0.15 * rng.standard_normal(200)
    out = p_spline(x, y, n_knots=18)
    mse = float(np.mean((np.asarray(out["fitted"]) - f) ** 2))
    assert mse < 0.03


def test_gcv_lambda_in_grid():
    rng = np.random.default_rng(1)
    x = np.sort(rng.random(150) * 4)
    y = np.cos(x) + 0.1 * rng.standard_normal(150)
    out = p_spline(x, y)
    assert 1e-4 <= float(out["lambda"]) <= 1e4


def test_fail_closed_mismatch():
    with pytest.raises(ValueError):
        p_spline(np.arange(10.0), np.arange(9.0))


def test_bench():
    out = bench_p_spline()
    assert out["synthetic_score"] == 1.0
