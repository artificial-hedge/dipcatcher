"""Tests for quant_fund.execution.sweep_bound."""

from __future__ import annotations

import pytest

from quant_fund.execution.sweep_bound import bound_bench, sweep_bound, sweep_bound_curve


def test_bound_within_top_level():
    levels = [(101.0, 50), (102.0, 50)]
    # 10 shares at best ask -> cost = 101, bound vs mid 100.5 -> 0.5
    assert sweep_bound(levels, 10, 100.5) == pytest.approx(0.5)


def test_bound_walks_levels():
    levels = [(101.0, 10), (102.0, 10)]
    # 20 shares: 10@101 + 10@102 -> mean 101.5 -> bound 1.0 vs mid 100.5
    assert sweep_bound(levels, 20, 100.5) == pytest.approx(1.0)


def test_unbounded_when_depth_insufficient():
    levels = [(101.0, 5)]
    assert sweep_bound(levels, 10, 100.0) is None


def test_bound_is_deterministic_worst_case():
    levels = [(101.0, 100), (103.0, 100), (107.0, 50)]
    b1 = sweep_bound(levels, 150, 100.0)
    b2 = sweep_bound(levels, 150, 100.0)
    assert b1 == b2 == pytest.approx((100 * 101 + 50 * 103) / 150 - 100.0)


def test_curve_shape_and_bad_input():
    levels = [(101.0, 50), (102.0, 50)]
    curve = sweep_bound_curve(levels, (10.0, 500.0), 100.0)
    assert curve[0]["bound_ticks"] is not None and curve[1]["bound_ticks"] is None
    with pytest.raises(ValueError):
        sweep_bound(levels, -1, 100.0)
    with pytest.raises(ValueError):
        sweep_bound(levels, 10, float("nan"))


def test_bench_sealed_and_deterministic():
    r1 = bound_bench(n_steps=4000)
    r2 = bound_bench(n_steps=4000)
    assert r1["schema"] == "sweep_bound.v1"
    assert r1["data_label"] == "SYNTHETIC"
    assert r1["live_pnl_claim"] is False
    assert r1 == r2
    assert "bounded_share" in r1["interpretation"]
