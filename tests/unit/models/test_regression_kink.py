"""Tests for regression kink design (models/regression_kink.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.regression_kink import (
    bench_regression_kink,
    regression_kink,
    synth_kink,
)


def _panel(**kw):
    return synth_kink(seed=26, **kw)


def test_fuzzy_recovers_effect():
    d = _panel(effect=0.5, kink_fs=1.0)
    out = regression_kink(
        np.asarray(d["running"]),
        np.asarray(d["outcome"]),
        treatment_slope=np.asarray(d["treatment"]),
    )
    assert abs(float(out["b_rk"]) - 0.5) < 0.3


def test_first_stage_kink_detected():
    d = _panel(kink_fs=1.0)
    out = regression_kink(
        np.asarray(d["running"]),
        np.asarray(d["outcome"]),
        treatment_slope=np.asarray(d["treatment"]),
    )
    assert float(out["fs_d_slope"]) > 0.4


def test_sharp_mode_no_treatment():
    d = _panel()
    out = regression_kink(np.asarray(d["running"]), np.asarray(d["outcome"]))
    assert "b_rk" not in out
    assert math.isfinite(out["d_slope"])


def test_no_kink_raises_or_weak():
    d = _panel(kink_fs=0.0)
    try:
        out = regression_kink(
            np.asarray(d["running"]),
            np.asarray(d["outcome"]),
            treatment_slope=np.asarray(d["treatment"]),
        )
        assert abs(float(out["fs_d_slope"])) < 0.3
    except ValueError:
        pass  # honest fail-closed on unidentified ratio


def test_sides_respect_threshold():
    d = _panel()
    out = regression_kink(np.asarray(d["running"]), np.asarray(d["outcome"]))
    assert out["n_below"] > 0 and out["n_above"] > 0


def test_validation():
    d = _panel()
    x = np.asarray(d["running"])
    y = np.asarray(d["outcome"])
    t_ = np.asarray(d["treatment"])
    with pytest.raises(ValueError):
        regression_kink(x[:10], y[:10])
    with pytest.raises(ValueError):
        regression_kink(np.abs(x), y)  # no values below 0
    with pytest.raises(ValueError):
        regression_kink(x, y, treatment_slope=t_[:5])
    y2 = y.copy()
    y2[0] = np.nan
    with pytest.raises(ValueError):
        regression_kink(x, y2)


def test_determinism():
    d = _panel()
    a = regression_kink(
        np.asarray(d["running"]),
        np.asarray(d["outcome"]),
        treatment_slope=np.asarray(d["treatment"]),
    )
    b = regression_kink(
        np.asarray(d["running"]),
        np.asarray(d["outcome"]),
        treatment_slope=np.asarray(d["treatment"]),
    )
    assert float(a["b_rk"]) == float(b["b_rk"])


def test_bench_keys():
    out = bench_regression_kink()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
