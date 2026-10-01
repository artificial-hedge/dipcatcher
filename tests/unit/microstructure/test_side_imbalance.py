"""Tests for side_imbalance: buy-vs-sell asymmetry on real vs sim."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from quant_fund.microstructure.side_imbalance import (
    _imbalance_stats,
    lobster_side_imbalance,
    side_imbalance_bench,
    sim_side_imbalance,
)


def test_stats_buy_skew() -> None:
    rng = np.random.default_rng(0)
    n = 500
    dirs = np.where(rng.uniform(size=n) < 0.7, 1, -1)
    sizes = np.full(n, 10.0)
    times = np.linspace(0, 100, n)
    subs = np.where(rng.uniform(size=n) < 0.6, 1, -1)
    imb = np.full(n, -0.2)  # ask side heavier
    out = _imbalance_stats(dirs, sizes, times, subs, imb, 0.0, 100.0)
    assert out["ok"]
    assert out["exec_buy_share"] == pytest.approx(0.7, abs=0.05)
    assert out["submission_buy_share"] == pytest.approx(0.6, abs=0.05)
    assert out["touch_imbalance_mean"] == pytest.approx(-0.2)


def test_stats_few() -> None:
    out = _imbalance_stats(
        np.ones(10), np.ones(10), np.arange(10.0), np.ones(10), np.ones(10), 0.0, 1.0
    )
    assert out["ok"] is False


def test_lobster_csv(tmp_path) -> None:
    msg = tmp_path / "m.csv"
    ob = tmp_path / "o.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        t = 34200.0
        for i in range(1, 200):
            t += 0.01
            # 3 buy-initiated execs (resting sell, direction -1) per sell
            if i % 4 == 0:
                wm.writerow([t, 4, i, 10, 4000, 1])  # resting buy hit
            else:
                wm.writerow([t, 4, i, 10, 4100, -1])  # resting sell lifted
            wo.writerow([4100, 100, 4000, 500])
    out = lobster_side_imbalance(msg, ob)
    assert out["ok"]
    assert out["exec_buy_share"] == pytest.approx(0.75, abs=0.02)
    assert out["touch_imbalance_mean"] == pytest.approx((500 - 100) / 600)


def test_sim_symmetric() -> None:
    out = sim_side_imbalance(horizon=8000, seed=3)
    assert out["ok"]
    assert out["exec_buy_share"] == pytest.approx(0.5, abs=0.1)


def test_missing_tape(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        side_imbalance_bench(tmp_path)
