"""Tests for interrupted time series (models/interrupted_ts.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.interrupted_ts import (
    bench_interrupted_ts,
    comparative_its,
    interrupted_ts,
    synth_its,
)


def test_level_detected():
    y = synth_its(level_shift=0.8, seed=37)
    out = interrupted_ts(y, tau=120)
    assert float(out["p_level"]) < 0.05


def test_slope_detected():
    y = synth_its(level_shift=0.0, slope_shift=0.03, seed=37)
    out = interrupted_ts(y, tau=120)
    assert float(out["p_slope"]) < 0.05


def test_no_break_keeps_null():
    y = synth_its(level_shift=0.0, slope_shift=0.0, seed=37)
    out = interrupted_ts(y, tau=120)
    assert float(out["p_level"]) > 0.05
    assert float(out["p_slope"]) > 0.05


def test_residual_autocorr_flagged():
    y = synth_its(ar=0.6, seed=37)
    out = interrupted_ts(y, tau=120)
    assert float(out["dw"]) < 1.5


def test_comparative_its():
    y_t = synth_its(level_shift=0.8, seed=37)
    y_c = synth_its(level_shift=0.0, seed=38)
    out = comparative_its(y_t, y_c, tau=120)
    assert float(out["p_level"]) < 0.1


def test_validation():
    y = synth_its(seed=37)
    with pytest.raises(ValueError):
        interrupted_ts(y[:40], tau=20)
    with pytest.raises(ValueError):
        interrupted_ts(y, tau=5)  # tau too early
    with pytest.raises(ValueError):
        interrupted_ts(y, tau=len(y) - 3)  # tau too late
    with pytest.raises(ValueError):
        interrupted_ts(y * 0.0, tau=120)  # constant
    bad = y.copy()
    bad[10] = np.nan
    with pytest.raises(ValueError):
        interrupted_ts(bad, tau=120)


def test_determinism():
    y = synth_its(seed=37)
    a = interrupted_ts(y, tau=120)
    b = interrupted_ts(y, tau=120)
    assert float(a["beta_level"]) == float(b["beta_level"])


def test_bench_keys():
    out = bench_interrupted_ts()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
