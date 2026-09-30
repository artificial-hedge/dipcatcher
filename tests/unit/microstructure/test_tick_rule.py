from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.tick_rule import (
    _classify,
    _error_stats,
    lobster_sign_errors,
    sim_sign_errors,
    tick_rule_bench,
)


def test_classify_quote_rule() -> None:
    # prices above mid -> buy; at mid -> tick fallback
    prices = [100.0, 100.0, 101.0, 100.0]
    mids = [99.5, 100.0, 100.5, 100.5]
    qs, ts = _classify(prices, mids)
    assert qs[0] == 1  # above mid
    assert ts[0] == 1  # forced first diff (p - (p-1)) is positive
    assert qs[3] == -1  # below mid


def test_error_stats_counts() -> None:
    true = [1, 1, -1, -1]
    pred = [1, -1, 0, -1]  # one wrong, one undecided
    out = _error_stats(true, pred, "x")
    assert out["undecided"] == 1
    assert out["misclass_rate"] == pytest.approx(0.3333)


def test_lobster_sign_errors_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        wm.writerow([34200.0, 1, 1, 100, 2500, 1])
        wo.writerow([4000, 100, 3000, 100])
        # exec at ask: resting order is a sell (direction=-1) -> aggressor buy
        wm.writerow([34200.5, 4, 1, 100, 4000, -1])
        wo.writerow([4000, 100, 3000, 100])
        # exec at bid: resting buy (direction=1) -> aggressor sell
        wm.writerow([34201.0, 4, 2, 100, 3000, 1])
        wo.writerow([4000, 100, 3000, 100])
    out = lobster_sign_errors(tmp_path)
    assert out["n_execs"] == 2
    # quote rule: exec at 4000 > mid 3500 -> buy (correct); exec at
    # 3000 < mid -> sell (correct)
    assert out["quote_rule"]["misclass_rate"] == 0.0


def test_sim_sign_errors_runs() -> None:
    out = sim_sign_errors(horizon=2000, seed=3)
    assert out["n_execs"] > 0
    assert 0.0 <= (out["quote_rule"]["misclass_rate"] or 0.0) <= 1.0


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        tick_rule_bench(tmp_path)
