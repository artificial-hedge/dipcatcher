"""Tests for fractional response models (models/fractional_response.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.fractional_response import (
    bench_fractional_response,
    fractional_fit,
    synth_fractional,
)


def test_beta_recovered():
    d = synth_fractional(beta=0.9, seed=37)
    out = fractional_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert abs(float(out["beta_1"]) - 0.9) < 0.2


def test_predictions_inbounds():
    d = synth_fractional(beta=1.2, seed=37)
    out = fractional_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert float(out["pred_inbounds"]) == 1.0


def test_significant():
    d = synth_fractional(beta=0.9, seed=37)
    out = fractional_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert float(out["p_1"]) < 0.05


def test_null_nonsignificant():
    d = synth_fractional(beta=0.0, seed=37)
    out = fractional_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert abs(float(out["beta_1"])) < 0.2


def test_calibration():
    d = synth_fractional(beta=0.9, seed=37)
    out = fractional_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert float(out["calibration"]) < 0.05


def test_boundary_mass_ok():
    rng = np.random.default_rng(37)
    y = np.concatenate([np.zeros(50), rng.uniform(0.1, 0.9, 150)])
    x = rng.normal(0, 1, 200)
    out = fractional_fit(y, x)
    assert math.isfinite(float(out["beta_1"]))


def test_validation():
    d = synth_fractional(seed=37)
    y, x = np.asarray(d["y"]), np.asarray(d["x"])
    with pytest.raises(ValueError):
        fractional_fit(y[:20], x[:20])
    y_bad = y.copy()
    y_bad[0] = 1.5
    with pytest.raises(ValueError):
        fractional_fit(y_bad, x)
    with pytest.raises(ValueError):
        fractional_fit(np.full(50, 0.5), np.random.default_rng(0).normal(size=50))
    y_nan = y.copy()
    y_nan[0] = np.nan
    with pytest.raises(ValueError):
        fractional_fit(y_nan, x)


def test_determinism():
    d = synth_fractional(seed=37)
    a = fractional_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    b = fractional_fit(np.asarray(d["y"]), np.asarray(d["x"]))
    assert float(a["beta_1"]) == float(b["beta_1"])


def test_bench_keys():
    out = bench_fractional_response()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
