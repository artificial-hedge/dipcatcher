"""Quandt-Andrews / Nyblom tests."""

from __future__ import annotations

import numpy as np

from quant_fund.models.quandt_andrews import (
    andrews_break_test,
    bench_qa,
    nyblom_test,
)


def _xy(seed=0, n=200, break_frac=None):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    if break_frac is None:
        y = 1.0 * x + rng.normal(0, 1, n)
    else:
        bp = int(break_frac * n)
        y = np.where(np.arange(n) < bp, 0.3 * x, 2.0 * x) + rng.normal(0, 1, n)
    X = np.column_stack([np.ones(n), x])
    return y, X


def test_no_break_not_rejected():
    y, X = _xy()
    res = andrews_break_test(y, X)
    assert res["sup_pvalue"] > 0.01


def test_break_detected():
    y, X = _xy(break_frac=0.6)
    res = andrews_break_test(y, X)
    assert res["sup_pvalue"] < 0.05
    assert abs(res["break_frac"] - 0.6) < 0.15


def test_break_at_edges_not_found():
    # trim .15: a break at 5% outside scan range should
    # give weaker evidence than mid-sample break
    y, X = _xy(break_frac=0.05)
    res = andrews_break_test(y, X)
    y2, X2 = _xy(break_frac=0.5)
    res2 = andrews_break_test(y2, X2)
    assert res["sup_wald"] < res2["sup_wald"]


def test_nyblom_detects_break():
    y, X = _xy(break_frac=0.5)
    r = nyblom_test(y, X)
    assert r["stat"] > r["crit5"]


def test_nyblom_flat_not_reject():
    y, X = _xy()
    r = nyblom_test(y, X)
    assert r["stat"] < r["crit5"] * 3


def test_bench_qa():
    out = bench_qa()
    assert out["synthetic_sup_pvalue"] < 0.05
