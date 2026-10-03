"""Tests for distribution regression (models/distribution_regression.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.distribution_regression import (
    bench_distribution_regression,
    distribution_regression,
    synth_dist_reg,
)


def _panel(**kw):
    return synth_dist_reg(seed=30, **kw)


def _fit(d):
    return distribution_regression(np.asarray(d["y"]), np.asarray(d["x"]))


def test_hetero_flattened_beta_path():
    out = _fit(_panel(hetero=True))
    out0 = _fit(_panel(hetero=False))
    assert float(out["mean_abs_beta1"]) < 0.75 * float(out0["mean_abs_beta1"])


def test_cdf_monotone():
    out = _fit(_panel())
    assert float(out["cdf_monotone"]) == 1.0


def test_median_near_zero():
    out = _fit(_panel())
    assert abs(float(out["median_at_ref"])) < 0.5


def test_threshold_path_shape():
    out = _fit(_panel())
    taus = np.asarray(out["taus"])
    betas = np.asarray(out["betas"])
    assert betas.shape == (taus.size, 2)


def test_validation():
    d = _panel()
    y, x = np.asarray(d["y"]), np.asarray(d["x"])
    with pytest.raises(ValueError):
        distribution_regression(y[:10], x[:10])
    with pytest.raises(ValueError):
        distribution_regression(y, x, link="probit")
    with pytest.raises(ValueError):
        distribution_regression(y, x, thresholds=np.array([0.0, 0.0, 1.0]))
    with pytest.raises(ValueError):
        distribution_regression(y, x, thresholds=np.array([100.0] * 6))
    y2 = y.copy()
    y2[0] = np.nan
    with pytest.raises(ValueError):
        distribution_regression(y2, x)


def test_determinism():
    a, b = _fit(_panel()), _fit(_panel())
    assert float(a["grad_var"]) == float(b["grad_var"])


def test_bench_keys():
    out = bench_distribution_regression()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
