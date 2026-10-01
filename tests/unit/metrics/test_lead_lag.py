"""Tests for Hayashi-Yoshida lead-lag (metrics/lead_lag.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.metrics.lead_lag import (
    bench_lead_lag,
    hy_covariance,
    lead_lag_scan,
    synth_async,
)


def test_lag_direction():
    d = synth_async(lag=2.0, seed=37)
    out = lead_lag_scan(
        np.asarray(d["t1"]),
        np.asarray(d["p1"]),
        np.asarray(d["t2"]),
        np.asarray(d["p2"]),
    )
    assert float(out["delta_star"]) > 0.5


def test_zero_lag():
    d = synth_async(lag=0.0, seed=37)
    out = lead_lag_scan(
        np.asarray(d["t1"]),
        np.asarray(d["p1"]),
        np.asarray(d["t2"]),
        np.asarray(d["p2"]),
    )
    assert abs(float(out["delta_star"])) < 2.0


def test_shift_improves_cov():
    d = synth_async(lag=3.0, seed=37)
    out = lead_lag_scan(
        np.asarray(d["t1"]),
        np.asarray(d["p1"]),
        np.asarray(d["t2"]),
        np.asarray(d["p2"]),
        max_shift=8.0,
    )
    assert float(out["contrast"]) > 0.0


def test_hy_positive_when_aligned():
    d = synth_async(lag=0.0, rho=0.9, seed=37)
    cov = hy_covariance(
        np.asarray(d["t1"]),
        np.asarray(d["p1"]),
        np.asarray(d["t2"]),
        np.asarray(d["p2"]),
    )
    assert cov > 0.0


def test_validation():
    d = synth_async(seed=37)
    t1, p1 = np.asarray(d["t1"]), np.asarray(d["p1"])
    t2, p2 = np.asarray(d["t2"]), np.asarray(d["p2"])
    with pytest.raises(ValueError):
        hy_covariance(t1[:10], p1[:10], t2, p2)
    with pytest.raises(ValueError):
        hy_covariance(t1[::-1], p1[::-1], t2, p2)  # unsorted
    with pytest.raises(ValueError):
        hy_covariance(t1, p1[:-3], t2, p2)  # length mismatch
    bad = p1.copy()
    bad[5] = np.nan
    with pytest.raises(ValueError):
        hy_covariance(t1, bad, t2, p2)


def test_determinism():
    d = synth_async(seed=37)
    a = lead_lag_scan(
        np.asarray(d["t1"]),
        np.asarray(d["p1"]),
        np.asarray(d["t2"]),
        np.asarray(d["p2"]),
    )
    b = lead_lag_scan(
        np.asarray(d["t1"]),
        np.asarray(d["p1"]),
        np.asarray(d["t2"]),
        np.asarray(d["p2"]),
    )
    assert float(a["delta_star"]) == float(b["delta_star"])


def test_bench_keys():
    out = bench_lead_lag()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
