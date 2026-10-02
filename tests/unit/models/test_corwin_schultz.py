"""Spread estimator tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.corwin_schultz import (
    amihud_illiq,
    bench_spread,
    corwin_schultz_spread,
    roll_spread,
)


def test_cs_on_bounce_series():
    rng = np.random.default_rng(0)
    n = 200
    mid = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    sign = rng.choice([-1.0, 1.0], n)
    obs = mid * (1 + sign * 0.004)
    h = np.maximum(obs, mid) * 1.005
    lo = np.minimum(obs, mid) * 0.995
    cs = corwin_schultz_spread(h, lo, obs)
    assert cs.size == n - 1
    assert (cs >= 0).all()


def test_roll_positive_on_bounce():
    rng = np.random.default_rng(0)
    mid = np.cumsum(rng.normal(0, 0.01, 300))
    sign = rng.choice([-1.0, 1.0], 300)
    obs = mid + sign * 0.004
    s = roll_spread(np.diff(obs))
    assert s > 0


def test_roll_zero_iid():
    rng = np.random.default_rng(1)
    s = roll_spread(rng.normal(0, 0.01, 500))
    assert s >= 0.0


def test_amihud_scaling():
    r = np.abs(np.random.default_rng(0).normal(0, 0.02, 50)) + 0.01
    v = np.full(50, 1e6)
    assert amihud_illiq(r, v) == pytest.approx(r.mean(), rel=1e-6)


def test_bench_spread():
    out = bench_spread()
    assert 0 < out["synthetic_cs_spread_mean"] < 0.05
