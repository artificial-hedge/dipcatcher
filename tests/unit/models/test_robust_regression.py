"""Robust regression: Huber/S/LTS/MM under contamination."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.robust_regression import (
    bench_robust,
    huber_irls,
    lts,
    mm_regression,
    s_estimator,
)


@pytest.fixture()
def contaminated() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    n = 80
    xv = rng.uniform(-2, 2, n)
    yv = 1.0 + 2.0 * xv + rng.normal(scale=0.2, size=n)
    yv[rng.choice(n, n // 5, replace=False)] += 5.0
    return np.column_stack([np.ones(n), xv]), yv


def test_huber_slope(contaminated):
    x, y = contaminated
    b = np.asarray(huber_irls(x, y)["coef"])
    assert abs(b[1] - 2.0) < 0.15


def test_s_estimator_slope(contaminated):
    x, y = contaminated
    b = np.asarray(s_estimator(x, y, seed=3)["coef"])
    assert abs(b[1] - 2.0) < 0.15


def test_lts_slope(contaminated):
    x, y = contaminated
    b = np.asarray(lts(x, y, seed=3)["coef"])
    assert abs(b[1] - 2.0) < 0.15


def test_mm_slope(contaminated):
    x, y = contaminated
    b = np.asarray(mm_regression(x, y, seed=3)["coef"])
    assert abs(b[1] - 2.0) < 0.15


def test_huber_clean_data():
    rng = np.random.default_rng(9)
    x = np.column_stack([np.ones(60), rng.uniform(-1, 1, 60)])
    y = 0.5 - 1.5 * x[:, 1] + rng.normal(scale=0.1, size=60)
    b = np.asarray(huber_irls(x, y)["coef"])
    assert abs(b[0] - 0.5) < 0.2
    assert abs(b[1] + 1.5) < 0.2


def test_bench_robust():
    out = bench_robust(seed=537)
    assert out["synthetic_mm_slope_err"] < 0.1
    assert out["synthetic_ols_slope_err"] > 0.05
