"""Tests for microstructure/trade_decomp.py — effective/realized/impact."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.trade_decomp import (
    _decomp_stats,
    lobster_trade_decomp,
    sim_trade_decomp,
    trade_decomp_bench,
)


def test_decomp_identity_math():
    # mid at t=0: 100.0; exec buy at 100.4 (eff=0.8); mid at +1s: 100.2
    execs = [(0.5, 1, 100.4)]
    mid_times = np.array([0.0, 0.4, 1.5])
    mids = np.array([100.0, 100.0, 100.2])
    out = _decomp_stats(execs, mid_times, mids)
    h = out["per_horizon"]["1.0s"]
    # pre-event mid = row index 1 (t=0.4 < 0.5) -> m0 = 100.0
    assert h["effective_mean"] == pytest.approx(0.8)
    # post: mid_times[2]=1.5 >= 0.4+1.0 -> mid=100.2; realized = 2*(100.4-100.2)=0.4
    assert h["realized_mean"] == pytest.approx(0.4)
    assert h["impact_mean"] == pytest.approx(0.4)
    assert h["impact_share"] == pytest.approx(0.5)


def test_sell_side_sign_flip():
    # sell aggressor at bid 99.6, mid 100.0 -> eff = 2*(-1)*(99.6-100) = 0.8
    execs = [(0.5, -1, 99.6)]
    mid_times = np.array([0.0, 0.4, 1.5])
    mids = np.array([100.0, 100.0, 100.0])
    out = _decomp_stats(execs, mid_times, mids)
    h = out["per_horizon"]["1.0s"]
    assert h["effective_mean"] == pytest.approx(0.8)
    assert h["realized_mean"] == pytest.approx(0.8)  # flat mid -> maker keeps all


def test_exec_before_first_mid_skipped():
    execs = [(0.0, 1, 100.4)]
    mid_times = np.array([1.0, 2.0])
    mids = np.array([100.0, 100.0])
    out = _decomp_stats(execs, mid_times, mids)
    assert out["per_horizon"].get("1.0s") is None
    assert np.isnan(out["effective_mean_ticks"]) or out["n_execs"] == 1


def test_sim_side_runs():
    out = sim_trade_decomp(None, seed=0, horizon=3000)
    assert out["n_execs"] >= 0
    assert "per_horizon" in out


def test_lobster_missing_tape_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        trade_decomp_bench(tmp_path)


def test_lobster_fixture(tmp_path):
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    # submit sell 1 @10000 (resting ask), then buy-exec it: direction of the
    # resting order is -1 (sell) so aggressor = +1 buy. Price ints /1e4.
    msg.write_text("1.0,1,1001,100,1000000,-1\n2.0,4,1001,50,1000000,-1\n")
    # two ob rows, 10 levels each, interleaved (ask_px,ask_sz,bid_px,bid_sz)
    row = ",".join(
        ["1000000", "10", "990000", "10", "1010000", "10", "980000", "10"]
        + ["0", "0", "0", "0"] * 8
    )
    ob.write_text(row + "\n" + row + "\n")
    out = lobster_trade_decomp(msg, ob)
    h = out["per_horizon"].get("1.0s")
    # exec at t=2.0: pre mid = row0 mid = (1000000+990000)/200 = 9950 ticks;
    # buy at 10000 ticks -> eff = 2*(50) = 100 ticks ($1 spread)
    assert h is not None and h["effective_mean"] == pytest.approx(100.0)
