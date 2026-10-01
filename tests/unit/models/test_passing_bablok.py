"""Passing-Bablok and Deming regression tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.passing_bablok import (
    bench_passing_bablok,
    deming_regression,
    passing_bablok,
)


def test_pb_recovers_linear_slope():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 150)
    y = 1.0 + 1.4 * x + rng.normal(0, 0.3, 150)
    fit = passing_bablok(x, y)
    assert abs(fit["slope"] - 1.4) < 0.08
    assert abs(fit["intercept"] - 1.0) < 0.3


def test_pb_robust_to_outliers():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, 200)
    y = 2.0 + 0.9 * x + rng.normal(0, 0.2, 200)
    idx = rng.choice(200, 10, replace=False)
    y[idx] += 15.0
    fit = passing_bablok(x, y)
    ols = float(np.polyfit(x, y, 1)[0])
    assert abs(fit["slope"] - 0.9) < abs(ols - 0.9)


def test_pb_slope_ci_brackets():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 10, 120)
    y = 0.5 * x + rng.normal(0, 0.4, 120)
    fit = passing_bablok(x, y)
    assert fit["slope_lo"] <= fit["slope"] <= fit["slope_hi"]


def test_deming_balanced_errors_is_tls():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 5, 150)
    y = 2.0 + 1.8 * x + rng.normal(0, 0.3, 150)
    fit = deming_regression(x, y, lam=1.0)
    assert abs(fit["slope"] - 1.8) < 0.1
    assert fit["se"] > 0


def test_input_validation():
    with pytest.raises(ValueError):
        passing_bablok(np.arange(5.0), np.arange(5.0))
    with pytest.raises(ValueError):
        deming_regression(np.arange(20.0), np.arange(20.0), lam=0.0)


def test_bench_passes():
    out = bench_passing_bablok()
    assert out["synthetic_pb_err"] < 0.15
    assert out["synthetic_pb_beats_ols"] == 1.0
    assert out["synthetic_runs_z"] < 2.5
    assert out["synthetic_deming_err"] < 0.15
    assert out["synthetic_score"] == 1.0
