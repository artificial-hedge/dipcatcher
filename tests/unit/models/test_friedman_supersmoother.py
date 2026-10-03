"""Friedman supersmoother tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.friedman_supersmoother import (
    bench_supersmoother,
    lowess,
    running_line,
    supersmoother,
)


def test_running_line_exact_on_linear():
    x = np.linspace(0, 1, 100)
    y = 2 + 3 * x
    s = running_line(x, y, 21)
    assert np.allclose(s, y, atol=1e-8)


def test_running_line_smooths_noise():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 100)
    y = rng.normal(0, 1, 100)
    s = running_line(x, y, 51)
    assert np.var(s) < np.var(y)


def test_lowess_recovers_smooth():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 100)
    f = np.sin(6 * x)
    y = f + rng.normal(0, 0.2, 100)
    s = lowess(x, y, 0.3)
    assert np.corrcoef(s, f)[0, 1] > 0.9


def test_supersmoother_output_shape():
    rng = np.random.default_rng(2)
    x = np.linspace(0, 1, 150)
    y = rng.normal(0, 1, 150)
    s = supersmoother(x, y)
    assert s.shape == y.shape and np.isfinite(s).all()


def test_supersmoother_adapts_to_curvature():
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 300)
    f = np.where(x < 0.5, np.sin(30 * np.pi * x), 0.5)
    y = f + rng.normal(0, 0.2, 300)
    ss = supersmoother(x, y)
    coarse = running_line(x, y, 150)
    # adaptive should beat coarse on the wiggly half
    fast = x < 0.5
    assert ((ss[fast] - f[fast]) ** 2).mean() < ((coarse[fast] - f[fast]) ** 2).mean()


def test_bench_supersmoother():
    out = bench_supersmoother()
    assert out["synthetic_mse_supersmoother"] < out["synthetic_mse_span50"]
