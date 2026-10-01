"""Tests for kernel regression (models/kernel_regression.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.kernel_regression import (
    bench_kernel_regression,
    local_linear,
    loo_cv_bandwidth,
    nadaraya_watson,
    synth_smooth_curve,
)


def _panel(**kw):
    return synth_smooth_curve(seed=35, **kw)


def test_local_linear_tracks_curve():
    d = _panel(nonlinear=True)
    x, y = np.asarray(d["x"]), np.asarray(d["y"])
    xe = np.linspace(-1.5, 1.5, 30)
    f, _ = local_linear(x, y, xe, 0.3)
    m_true = np.sin(2 * xe) * (1 + 0.3 * xe)
    assert np.sqrt(np.mean((f - m_true) ** 2)) < 0.35


def test_derivative_estimated():
    d = _panel(nonlinear=True)
    x, y = np.asarray(d["x"]), np.asarray(d["y"])
    xe = np.linspace(-1.5, 1.5, 30)
    _, dv = local_linear(x, y, xe, 0.3)
    dm = 2 * np.cos(2 * xe) * (1 + 0.3 * xe) + 0.3 * np.sin(2 * xe)
    assert np.corrcoef(dv, dm)[0, 1] > 0.5


def test_linear_curve_fit():
    d = _panel(nonlinear=False)
    x, y = np.asarray(d["x"]), np.asarray(d["y"])
    xe = np.linspace(-1.0, 1.0, 20)
    f, _ = local_linear(x, y, xe, 0.5)
    assert np.sqrt(np.mean((f - 0.5 * xe) ** 2)) < 0.25


def test_cv_bandwidth_positive():
    d = _panel(nonlinear=True)
    bw = loo_cv_bandwidth(np.asarray(d["x"]), np.asarray(d["y"]))
    assert bw > 0.05


def test_nw_shape():
    d = _panel()
    f = nadaraya_watson(np.asarray(d["x"]), np.asarray(d["y"]), np.array([0.0, 1.0]), 0.4)
    assert f.shape == (2,) and np.all(np.isfinite(f))


def test_validation():
    d = _panel()
    x, y = np.asarray(d["x"]), np.asarray(d["y"])
    with pytest.raises(ValueError):
        local_linear(x[:10], y[:10], np.array([0.0]), 0.3)
    with pytest.raises(ValueError):
        local_linear(x, y, np.array([0.0]), -0.1)
    with pytest.raises(ValueError):
        nadaraya_watson(x, y, np.array([0.0]), 0.0)
    with pytest.raises(ValueError):
        nadaraya_watson(x, y, np.array([1e6]), 0.4)
    y2 = y.copy()
    y2[0] = np.nan
    with pytest.raises(ValueError):
        local_linear(x, y2, np.array([0.0]), 0.3)


def test_determinism():
    d = _panel()
    a, _ = local_linear(np.asarray(d["x"]), np.asarray(d["y"]), np.array([0.5]), 0.3)
    b, _ = local_linear(np.asarray(d["x"]), np.asarray(d["y"]), np.array([0.5]), 0.3)
    assert float(a[0]) == float(b[0])


def test_bench_keys():
    out = bench_kernel_regression()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
