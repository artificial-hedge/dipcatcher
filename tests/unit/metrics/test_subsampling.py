"""Tests for Politis-Romano-Wolf subsampling (metrics/subsampling.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.subsampling import (
    _ar1_mean,
    _ar1_sampler,
    bench_subsampling,
    subsampling_ci,
    subsampling_coverage,
)


def _ar1(rho: float, t: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, 1.0, t)
    y = np.zeros(t)
    for i in range(1, t):
        y[i] = rho * y[i - 1] + e[i]
    return y.reshape(-1, 1)


def test_ci_brackets_truth():
    x = _ar1(0.6, 400, 37)
    out = subsampling_ci(_ar1_mean, x, level=0.9, seed=37)
    assert float(out["ci_lo"]) <= 0.6 <= float(out["ci_hi"])


def test_theta_close_to_truth():
    x = _ar1(0.6, 400, 37)
    out = subsampling_ci(_ar1_mean, x)
    assert abs(float(out["theta"]) - 0.6) < 0.1


def test_coverage_reasonable():
    cov = subsampling_coverage(_ar1_mean, _ar1_sampler(0.6, 300), 0.6, n_rep=40, seed=37)
    assert 0.6 <= float(cov["coverage"]) <= 1.0


def test_narrower_level_wider_width():
    x = _ar1(0.5, 300, 37)
    w80 = float(subsampling_ci(_ar1_mean, x, level=0.8)["ci_width"])
    w95 = float(subsampling_ci(_ar1_mean, x, level=0.95)["ci_width"])
    assert w95 > w80


def test_validation():
    x = _ar1(0.5, 300, 37)
    with pytest.raises(ValueError):
        subsampling_ci(_ar1_mean, x[:50])
    with pytest.raises(ValueError):
        subsampling_ci(_ar1_mean, x, level=0.5)
    with pytest.raises(ValueError):
        subsampling_ci(_ar1_mean, x, block_frac=0.9)
    with pytest.raises(ValueError):
        subsampling_ci(_ar1_mean, x.ravel())  # 1-d
    bad = x.copy()
    bad[10, 0] = np.nan
    with pytest.raises(ValueError):
        subsampling_ci(_ar1_mean, bad)


def test_determinism():
    x = _ar1(0.5, 300, 37)
    a = subsampling_ci(_ar1_mean, x, seed=37)
    b = subsampling_ci(_ar1_mean, x, seed=37)
    assert float(a["ci_lo"]) == float(b["ci_lo"])


def test_bench_keys():
    out = bench_subsampling()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
