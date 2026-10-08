"""Tests for calibration_survey — GREG weights."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.calibration_survey import (
    bench_calibration_survey,
    greg_weights,
    truncated_calibration,
)


def test_margins_exact():
    rng = np.random.default_rng(0)
    n = 200
    x = np.stack([rng.gamma(2.0, 1.0, n), rng.random(n)], axis=1)
    d = np.ones(n) * 5.0
    totals = np.array([1000.0, 400.0])
    w = greg_weights(x, d, totals)
    marg = (w[:, None] * x).sum(axis=0)
    assert np.abs(marg - totals).max() / totals.max() < 1e-6


def test_greg_improves_biased_sample():
    rng = np.random.default_rng(1)
    x_pop = rng.gamma(2.0, 1.0, 4000)
    y_pop = 3.0 * x_pop + rng.standard_normal(4000)
    pi = np.clip(300 * x_pop / x_pop.sum(), 1e-3, 0.9)
    sel = rng.random(4000) < pi
    w = greg_weights(x_pop[sel][:, None], 1.0 / pi[sel], np.array([x_pop.sum()]))
    est = float((w * y_pop[sel]).sum())
    assert abs(est - y_pop.sum()) / y_pop.sum() < 0.15


def test_truncated_bounds_weights():
    rng = np.random.default_rng(2)
    n = 150
    x = rng.gamma(2.0, 1.0, n)[:, None]
    d = np.ones(n)
    w = truncated_calibration(x, d, np.array([300.0]), lo=0.5, hi=3.0)
    assert (w / d <= 3.0 + 1e-9).all() and (w / d >= 0.5 - 1e-9).all()


def test_fail_closed_singular():
    x = np.stack([np.ones(50), np.ones(50)], axis=1)
    with pytest.raises(ValueError):
        greg_weights(x, np.ones(50), np.array([1.0, 2.0]))


def test_bench():
    out = bench_calibration_survey()
    assert out["synthetic_score"] == 1.0
