"""post_trade_drift contracts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.post_trade_drift import (
    EVENT_HORIZONS,
    _drift_kernel,
    lobster_post_trade_drift,
    post_trade_drift_bench,
    sim_post_trade_drift,
)


def test_kernel_continuation() -> None:
    # every exec followed by +1 tick drift over next 5 events
    mids = np.arange(2000, dtype=float)
    idx = np.arange(20, 1980, 5)
    out = _drift_kernel(mids, idx, np.ones(idx.size, dtype=int))
    assert out["ok"]
    k = out["all"]["kernel"]
    assert k["5"]["mean_ticks"] == pytest.approx(5.0)
    assert out["movers"]["n"] > 100


def test_kernel_reversion() -> None:
    # exec i pushes mid +2, then it reverts −2 over next 10 events
    mids = np.full(3000, 4000.0)
    idx = np.arange(10, 2800, 55)  # spacing must not collide with the horizons
    for i in idx:
        mids[i : i + 10] += 2.0
    out = _drift_kernel(mids, idx, np.ones(idx.size, dtype=int))
    assert out["ok"]
    # fully reverted 10 events later → drift = -2 at horizons >= 10
    assert out["all"]["kernel"]["50"]["mean_ticks"] == pytest.approx(-2.0)
    assert out["all"]["kernel"]["20"]["mean_ticks"] == pytest.approx(-2.0)
    assert out["all"]["kernel"]["1"]["mean_ticks"] == pytest.approx(0.0)


def test_kernel_too_few() -> None:
    out = _drift_kernel(np.arange(30.0), np.arange(5), np.ones(5, dtype=int))
    assert not out["ok"]


def test_lobster_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        t = 34200.0
        px = 4000
        for oid in range(1, 301):
            t += 0.01
            wm.writerow([t, 4, oid, 10, px, -1])
            px += 10  # mid ratchets up
            wo.writerow([px + 100, 500, px, 500])
    out = lobster_post_trade_drift(msg, ob)
    assert out["ok"] and out["all"]["n"] >= 250
    assert out["all"]["kernel"]["5"]["mean_ticks"] > 0.0


def test_sim_runs() -> None:
    out = sim_post_trade_drift(horizon=6000, seed=5)
    assert out["ok"] and out["all"]["n"] > 20
    for h in EVENT_HORIZONS:
        assert str(h) in out["all"]["kernel"]


def test_bench_missing_tape(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        post_trade_drift_bench(tmp_path)
