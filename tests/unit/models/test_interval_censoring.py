"""Tests for Turnbull interval-censored NPMLE (models/interval_censoring.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quant_fund.models.interval_censoring import (
    bench_interval_censoring,
    midpoint_naive_median,
    synth_interval,
    turnbull_fit,
)


def test_median_recovery():
    d = synth_interval(rate=0.5, seed=37)
    out = turnbull_fit(np.asarray(d["left"]), np.asarray(d["right"]))
    # true Exp(.5) median = ln2/.5
    assert abs(float(out["median"][0]) - math.log(2) / 0.5) < 0.45


def test_beats_midpoint():
    d = synth_interval(inspect_every=0.8, rate=0.5, seed=37)
    out = turnbull_fit(np.asarray(d["left"]), np.asarray(d["right"]))
    naive = midpoint_naive_median(np.asarray(d["left"]), np.asarray(d["right"]))
    true = math.log(2) / 0.5
    assert abs(float(out["median"][0]) - true) <= abs(naive - true) + 0.1


def test_mass_sums_one():
    d = synth_interval(seed=37)
    out = turnbull_fit(np.asarray(d["left"]), np.asarray(d["right"]))
    assert abs(float(np.sum(out["mass"])) - 1.0) < 1e-6
    assert np.all(out["mass"] >= -1e-12)


def test_survival_monotone():
    d = synth_interval(seed=37)
    out = turnbull_fit(np.asarray(d["left"]), np.asarray(d["right"]))
    s = np.asarray(out["surv_at_hi"])
    assert np.all(np.diff(s) <= 1e-9)


def test_narrow_intervals_accurate():
    d = synth_interval(inspect_every=0.2, rate=0.5, seed=37)
    out = turnbull_fit(np.asarray(d["left"]), np.asarray(d["right"]))
    assert abs(float(out["median"][0]) - math.log(2) / 0.5) < 0.4


def test_validation():
    d = synth_interval(seed=37)
    lo, hi = np.asarray(d["left"]), np.asarray(d["right"])
    with pytest.raises(ValueError):
        turnbull_fit(lo[:10], hi[:10])
    with pytest.raises(ValueError):
        turnbull_fit(hi, lo)  # swapped: right <= left
    l2 = lo.copy()
    l2[0] = -1
    with pytest.raises(ValueError):
        turnbull_fit(l2, hi)
    l3 = lo.copy()
    l3[0] = np.nan
    with pytest.raises(ValueError):
        turnbull_fit(l3, hi)


def test_determinism():
    d = synth_interval(seed=37)
    a = turnbull_fit(np.asarray(d["left"]), np.asarray(d["right"]))
    b = turnbull_fit(np.asarray(d["left"]), np.asarray(d["right"]))
    assert float(a["median"][0]) == float(b["median"][0])


def test_bench_keys():
    out = bench_interval_censoring()
    for k, v in out.items():
        assert k.startswith("synthetic_")
        assert math.isfinite(v)
    assert out["synthetic_detects"] == 1.0
    assert out["synthetic_determinism"] == 1.0
