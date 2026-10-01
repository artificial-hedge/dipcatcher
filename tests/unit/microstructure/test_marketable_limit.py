"""marketable_limit contracts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.marketable_limit import (
    _aggression_stats,
    lobster_marketable_limit,
    marketable_limit_bench,
    sim_marketable_limit,
)


def test_stats_buckets() -> None:
    rel = np.array([2.0, 1.0, 0.0, -1.0, -5.0] * 30)
    rel_own = np.array([5.0, 4.0, 3.0, 2.0, -2.0] * 30)
    sizes = np.full(rel.size, 10.0)
    out = _aggression_stats(rel, rel_own, sizes)
    assert out["ok"]
    assert out["share"]["marketable"] == pytest.approx(0.6)
    assert out["share"]["crossing"] == pytest.approx(0.4)
    assert out["share"]["inside"] == pytest.approx(0.2)
    assert out["share"]["at_own_touch"] == pytest.approx(0.0)
    assert out["share"]["behind"] == pytest.approx(0.2)


def test_stats_too_few() -> None:
    out = _aggression_stats(np.arange(10.0), np.arange(10.0), np.ones(10))
    assert not out["ok"]


def test_lobster_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        t = 34200.0
        for i in range(1, 201):
            t += 0.01
            # alternate: marketable buy (price >= best ask 4100) and resting
            px = 4100 if i % 2 == 0 else 3950
            wm.writerow([t, 1, i, 10, px, 1])
            wo.writerow([4100, 500, 4000, 500])
    out = lobster_marketable_limit(msg, ob)
    assert out["ok"]
    assert out["share"]["marketable"] == pytest.approx(0.5, abs=0.05)


def test_sim_no_marketable() -> None:
    out = sim_marketable_limit(horizon=4000, seed=5)
    assert out["ok"]
    assert out["share"]["marketable"] == 0.0


def test_bench_missing_tape(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        marketable_limit_bench(tmp_path)
