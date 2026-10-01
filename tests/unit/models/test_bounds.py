"""Tests for partial-identification bounds (models/bounds.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.bounds import (
    bench_bounds,
    lee_bounds,
    manski_bounds,
    manski_mar_bounds,
    synth_missing,
    synth_selection,
)


def test_manski_covers_truth():
    d = synth_missing(n=1200, mu=2.0, miss_frac=0.3, seed=9)
    y_obs = np.asarray(d["y_obs"])
    m = np.asarray(d["missing"])
    yc = np.where(m > 0.5, 0.0, y_obs)
    mb = manski_bounds(yc, m, y_lo=-3.0, y_hi=7.0)
    assert mb["lb"] <= 2.0 <= mb["ub"]
    assert mb["width"] == pytest.approx(mb["ub"] - mb["lb"])


def test_manski_width_scales_with_missing():
    base = synth_missing(n=800, seed=3)
    y = np.asarray(base["y"])
    z = np.zeros_like(y)
    mb0 = manski_bounds(y, z, y_lo=-3, y_hi=5)
    m_half = (np.arange(y.size) % 2).astype(np.float64)
    mb1 = manski_bounds(y, m_half, y_lo=-3, y_hi=5)
    assert mb1["width"] > mb0["width"]


def test_mar_point():
    d = synth_missing(n=2000, mu=2.0, miss_frac=0.4, seed=4)
    m = np.asarray(d["missing"])
    mar = manski_mar_bounds(np.asarray(d["y_obs"]).copy(), m)
    assert abs(mar["mar_mean"] - 2.0) < 0.2
    assert mar["mar_ci_lb"] < mar["mar_ci_ub"]


def test_lee_covers_tau():
    d = synth_selection(n=1500, tau=1.0, seed=11)
    lb = lee_bounds(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["observed"]))
    assert lb["lb"] <= 1.0 <= lb["ub"]
    assert lb["excess_share"] > 0.1


def test_lee_requires_more_selection():
    rng = np.random.default_rng(0)
    n = 400
    t = (rng.random(n) < 0.5).astype(np.float64)
    s = (rng.random(n) < 0.3 + 0.2 * (1 - t)).astype(np.float64)  # control selects more
    y = rng.standard_normal(n)
    with pytest.raises(ValueError):
        lee_bounds(y, t, s)


def test_lee_validation():
    with pytest.raises(ValueError):
        lee_bounds(np.ones(30), np.ones(30), np.ones(30))
    with pytest.raises(ValueError):
        lee_bounds(np.ones(100), np.ones(100), np.ones(50))


def test_manski_validation():
    with pytest.raises(ValueError):
        manski_bounds(np.ones(10), np.zeros(10))
    with pytest.raises(ValueError):
        manski_bounds(np.ones(100), np.ones(100))
    with pytest.raises(ValueError):
        manski_bounds(np.ones(50), np.zeros(50), y_lo=1.0, y_hi=1.0)


def test_determinism():
    d = synth_selection(n=800, seed=5)
    a = lee_bounds(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["observed"]))["lb"]
    b = lee_bounds(np.asarray(d["y"]), np.asarray(d["treat"]), np.asarray(d["observed"]))["lb"]
    assert a == b


def test_bench_keys():
    out = bench_bounds()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_manski_covers"] == 1.0
    assert out["synthetic_lee_covers"] == 1.0
    assert out["synthetic_mar_err"] < 0.2
    assert out["synthetic_determinism"] == 1.0
