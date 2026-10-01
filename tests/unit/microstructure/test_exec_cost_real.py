from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.exec_cost_real import (
    _bucket_stats,
    _size_cost_exponent,
    exec_cost_bench,
    lobster_exec_cost,
    sim_exec_cost,
)


def test_bucket_stats_groups() -> None:
    qty = np.asarray([50, 50, 50, 200, 200, 500, 500, 500, 3000, 3000, 3000, 3000])
    cost = np.asarray([0.1, 0.1, 0.1, 0.2, 0.2, 0.4, 0.4, 0.4, 1.0, 1.0, 1.0, 1.0])
    out = _bucket_stats(qty, cost)
    assert "qty_1_100" in out
    assert "qty_101_300" not in out  # n=2 below the min-count gate
    assert out["qty_301_1000"]["n"] == 3
    assert out["qty_1001_5000"]["mean_ticks_beyond_mid"] == pytest.approx(1.0)


def test_exponent_planted() -> None:
    rng = np.random.default_rng(0)
    qty = rng.lognormal(5, 1, 200)
    cost = 0.5 * np.sqrt(qty / 100) * np.exp(rng.normal(0, 0.05, 200))
    e = _size_cost_exponent(qty, cost)
    assert e is not None and 0.35 < e < 0.65


def test_exponent_degenerate() -> None:
    assert _size_cost_exponent(np.ones(10), np.ones(10)) is None


def test_lobster_exec_cost_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    rows = [
        # seed row: ask 5010 x 100, bid 4990 x 100
        (34200.0, 1, 1, 100, 5010, -1, [(5010, 100)], [(4990, 100)]),
        (34200.5, 1, 2, 100, 4990, 1, [(5010, 100)], [(4990, 100)]),
        # marketable buy exec: lifts the whole ask (100 @ 5010 = 10 ticks above mid 5000)
        (34201.0, 4, 1, 100, 5010, -1, [(5050, 10)], [(4990, 100)]),
    ]
    with msg.open("w", newline="") as f:
        for t, ty, oid, sz, px, d, *_ in rows:
            csv.writer(f).writerow([t, ty, oid, sz, px, d])
    with ob.open("w", newline="") as f:
        w = csv.writer(f)
        for *_, asks, bids in rows:
            row: list[str] = []
            for px, sz in asks:
                row += [str(px), str(sz)]
            for px, sz in bids:
                row += [str(px), str(sz)]
            w.writerow(row)
    out = lobster_exec_cost(tmp_path)
    assert out["n_execs"] == 1
    assert out["mean_ticks_beyond_mid"] == pytest.approx(0.1)  # 10 raw units = 0.1 ticks
    assert out["share_negative_fills"] == pytest.approx(0.0)


def test_sim_exec_cost_runs() -> None:
    out = sim_exec_cost(horizon=1500, seed=3)
    assert out["n_execs"] > 10
    assert out["mean_ticks_beyond_mid"] is not None


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        exec_cost_bench(tmp_path)
