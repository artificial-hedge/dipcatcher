"""Business-cycle filter tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.hamilton_filter import (
    bench_hamilton,
    bk_filter,
    cf_filter,
    hamilton_filter,
    hp_filter,
)


def _trend_cycle(seed=0, n=160):
    rng = np.random.default_rng(seed)
    trend = 0.4 * np.arange(n)
    cyc = np.zeros(n)
    eps = rng.normal(0, 1, n)
    for t in range(2, n):
        cyc[t] = 1.3 * cyc[t - 1] - 0.5 * cyc[t - 2] + eps[t]
    return trend + cyc + 50.0, cyc


def test_hamilton_cycle_stationary():
    y, cyc = _trend_cycle()
    _, c = hamilton_filter(y)
    v = ~np.isnan(c)
    assert np.corrcoef(c[v], cyc[v])[0, 1] > 0.5


def test_hp_identity():
    y, _ = _trend_cycle()
    tr, c = hp_filter(y, 1600)
    assert np.allclose(tr + c, y)
    # trend smoother than raw
    assert np.std(np.diff(tr, 2)) < np.std(np.diff(y, 2))


def test_bk_pads_nan():
    y, _ = _trend_cycle()
    c = bk_filter(y)
    assert np.isnan(c[:12]).all() and np.isnan(c[-12:]).all()


def test_cf_no_nan():
    y, cyc = _trend_cycle()
    c = cf_filter(y)
    assert np.isfinite(c).all()
    assert np.corrcoef(c[20:-20], cyc[20:-20])[0, 1] > 0.5


def test_bench_hamilton():
    out = bench_hamilton()
    assert out["synthetic_corr_hamilton"] > 0.7
