"""spread_response contracts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.spread_response import (
    HORIZONS_S,
    _response_kernel,
    lobster_spread_response,
    sim_spread_response,
)


def test_kernel_flat_stream_zero_response() -> None:
    # constant spread 2 ticks, execs every event → kernel = 0
    times = np.arange(2000, dtype=float) * 0.01
    spreads = np.full(2000, 2)
    idx = np.arange(2000)
    out = _response_kernel(times, spreads, idx, np.ones(2000, dtype=int))
    assert out["ok"] and out["baseline_spread"] == pytest.approx(2.0)
    assert out["by_dir"]["buy_initiated"]["kernel"]["1s"]["median_delta"] == pytest.approx(0.0)


def test_kernel_detects_widening() -> None:
    # after each exec the spread widens by +3 ticks for the next 10 events
    times = np.arange(4000, dtype=float) * 0.01
    spreads = np.full(4000, 2)
    exec_idx = np.arange(0, 4000, 40)
    for i in exec_idx:
        spreads[i + 1 : i + 11] = 5
    out = _response_kernel(times, spreads, exec_idx, np.ones(exec_idx.size, dtype=int))
    k = out["by_dir"]["pooled"]["kernel"]
    assert k["0.05s"]["median_delta"] == pytest.approx(3.0)
    assert k["0.05s"]["share_wider"] == pytest.approx(1.0)


def test_kernel_too_few_execs() -> None:
    out = _response_kernel(
        np.arange(100, dtype=float), np.full(100, 2), np.arange(10), np.ones(10, dtype=int)
    )
    assert not out["ok"]


def test_lobster_spread_response_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm = csv.writer(fm)
        wo = csv.writer(fo)
        t = 34200.0
        oid = 0
        for _ in range(300):
            t += 0.005
            oid += 1
            wm.writerow([t, 1, oid, 10, 4000, 1])  # bid submit
            wo.writerow([4100, 1000, 4000, 1000])  # ask10 @ 4100 / bid10 @ 4000
            oid += 1
            wm.writerow([t, 1, oid, 10, 4100, -1])  # ask submit
            wo.writerow([4100, 1000, 4000, 1000])
            t += 0.005
            oid += 1
            wm.writerow([t, 4, oid, 10, 4100, -1])  # exec hitting resting sell
            wo.writerow([4100, 990, 4000, 1000])
    out = lobster_spread_response(msg, ob)
    assert out["ok"] and out["by_dir"]["buy_initiated"]["n"] >= 200
    # exec price 4100 = resting sell hit → aggressor = buy (+1)
    assert out["by_dir"]["sell_initiated"]["ok"] is False


def test_sim_response_runs() -> None:
    out = sim_spread_response(horizon=6000, seed=5)
    assert out["ok"] and out["by_dir"]["pooled"]["n"] > 20
    for h in HORIZONS_S:
        assert f"{h:g}s" in out["by_dir"]["pooled"]["kernel"]
