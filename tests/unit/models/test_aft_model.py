"""Tests for Weibull AFT MLE (models/aft_model.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.aft_model import (
    aft_fit,
    bench_aft_model,
    synth_aft,
)


def test_beta_recovered():
    d = synth_aft(beta=-0.5, seed=37)
    out = aft_fit(np.asarray(d["t"]), np.asarray(d["d"]), np.asarray(d["x"]))
    assert abs(float(out["beta_1"]) + 0.5) < 0.15


def test_sigma_recovered():
    d = synth_aft(sigma=0.6, seed=37)
    out = aft_fit(np.asarray(d["t"]), np.asarray(d["d"]), np.asarray(d["x"]))
    assert abs(float(out["sigma"]) - 0.6) < 0.15


def test_significant():
    d = synth_aft(beta=-0.5, seed=37)
    out = aft_fit(np.asarray(d["t"]), np.asarray(d["d"]), np.asarray(d["x"]))
    assert float(out["p_1"]) < 0.05


def test_beats_naive_under_censoring():
    d = synth_aft(beta=-0.5, censor_rate=0.4, seed=37)
    out = aft_fit(np.asarray(d["t"]), np.asarray(d["d"]), np.asarray(d["x"]))
    assert abs(float(out["beta_1"]) + 0.5) <= abs(float(out["beta_naive_1"]) + 0.5)


def test_no_censoring_fine():
    d = synth_aft(beta=0.4, censor_rate=0.01, seed=37)
    out = aft_fit(np.asarray(d["t"]), np.asarray(d["d"]), np.asarray(d["x"]))
    assert abs(float(out["beta_1"]) - 0.4) < 0.15


def test_validation():
    d = synth_aft(seed=37)
    t, dd, x = np.asarray(d["t"]), np.asarray(d["d"]), np.asarray(d["x"])
    with pytest.raises(ValueError):
        aft_fit(t[:20], dd[:20], x[:20])
    with pytest.raises(ValueError):
        aft_fit(-t, dd, x)
    d_bad = dd.copy()
    d_bad[:] = 0
    with pytest.raises(ValueError):
        aft_fit(t, d_bad, x)
    t_nan = t.copy()
    t_nan[0] = np.nan
    with pytest.raises(ValueError):
        aft_fit(t_nan, dd, x)


def test_determinism():
    d = synth_aft(seed=37)
    a = aft_fit(np.asarray(d["t"]), np.asarray(d["d"]), np.asarray(d["x"]))
    b = aft_fit(np.asarray(d["t"]), np.asarray(d["d"]), np.asarray(d["x"]))
    assert float(a["beta_1"]) == float(b["beta_1"])


def test_bench_keys():
    out = bench_aft_model()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
