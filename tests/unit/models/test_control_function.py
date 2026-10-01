"""Tests for control-function estimation (models/control_function.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.control_function import (
    bench_control_function,
    control_function,
    synth_endogenous,
)


def _panel(**kw):
    return synth_endogenous(seed=34, **kw)


def _fit(d):
    return control_function(np.asarray(d["y"]), np.asarray(d["endog"]), None, np.asarray(d["z"]))


def test_cf_beats_ols():
    out = _fit(_panel(beta=1.0, endog_strength=0.5))
    assert abs(float(out["beta_cf"]) - 1.0) < abs(float(out["beta_ols"]) - 1.0)


def test_dwh_detects_endogeneity():
    out = _fit(_panel(endog_strength=0.5))
    assert float(out["dwh_p"]) < 0.05


def test_dwh_null():
    out = _fit(_panel(endog_strength=0.0))
    assert float(out["dwh_p"]) > 0.01


def test_first_stage_strong():
    out = _fit(_panel(pi=0.8))
    assert float(out["fs_f"]) > 10.0


def test_ols_bias_direction():
    out = _fit(_panel(endog_strength=0.5))
    assert float(out["beta_ols"]) > 1.1  # positive corr → upward bias


def test_validation():
    d = _panel()
    y, e, z = (
        np.asarray(d["y"]),
        np.asarray(d["endog"]),
        np.asarray(d["z"]),
    )
    with pytest.raises(ValueError):
        control_function(y[:20], e[:20], None, z[:20])
    with pytest.raises(ValueError):
        control_function(y, e[:10], None, z)
    y2 = y.copy()
    y2[0] = np.nan
    with pytest.raises(ValueError):
        control_function(y2, e, None, z)


def test_determinism():
    a, b = _fit(_panel()), _fit(_panel())
    assert float(a["beta_cf"]) == float(b["beta_cf"])


def test_bench_keys():
    out = bench_control_function()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
