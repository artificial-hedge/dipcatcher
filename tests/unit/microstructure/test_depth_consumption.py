from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.depth_consumption import (
    _consume_stats,
    depth_consumption_bench,
    lobster_depth_consumption,
    sim_depth_consumption,
)


def test_consume_stats() -> None:
    out = _consume_stats([0.2, 0.5, 1.0, 1.5])
    assert out["n_fills"] == 4
    assert out["full_sweep_share"] == pytest.approx(0.5)
    assert out["over_sweep_share"] == pytest.approx(0.25)


def test_consume_stats_empty() -> None:
    assert _consume_stats([])["ok"] is False


def test_lobster_depth_consumption_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        wm.writerow([34200.0, 1, 1, 100, 2500, 1])
        wo.writerow([4000, 200, 3000, 100])  # ask depth 200 @ 4000
        # buyer-initiated exec of 50 @ 4000 against sell-side resting (dir -1)
        wm.writerow([34201.0, 4, 9, 50, 4000, -1])
        wo.writerow([4000, 150, 3000, 100])
    out = lobster_depth_consumption(tmp_path)
    assert out["n_exec_total"] == 1
    assert out["n_with_depth"] == 1
    assert out["mean_consumption"] == pytest.approx(0.25)


def test_sim_depth_consumption_runs() -> None:
    out = sim_depth_consumption(horizon=1500, seed=3)
    assert out["mode"] == "qty_vs_maker_queue_at_submit"


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        depth_consumption_bench(tmp_path)
