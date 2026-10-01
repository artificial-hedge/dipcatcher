from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.spread_dynamics import (
    _spread_profile,
    lobster_spread_dynamics,
    sim_spread_dynamics,
    spread_dynamics_bench,
)


def test_spread_profile_basics() -> None:
    spreads = [1.0, 1.0, 2.0, 3.0, 13.0, 20.0, 7.0, 1.0]
    out = _spread_profile(spreads)
    assert out["median_spread_ticks"] == pytest.approx(2.5)
    assert out["tight_share_le2"] == pytest.approx(4 / 8)


def test_spread_profile_empty() -> None:
    assert _spread_profile([])["ok"] is False


def test_sim_spread_dynamics_runs() -> None:
    out = sim_spread_dynamics(horizon=1500, seed=3)
    assert out["n_obs"] > 100


def test_lobster_spread_dynamics_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm = csv.writer(fm)
        wo = csv.writer(fo)
        # seed row (SUBMISSION events after book snapshot semantics)
        wm.writerow([34200.0, 1, 1, 100, 2500, 1])
        wo.writerow([4000, 100, 3000, 100])
        wm.writerow([34200.5, 1, 2, 100, 2500, 1])
        wo.writerow([4000, 200, 3500, 100])
    out = lobster_spread_dynamics(tmp_path)
    assert out["n_obs"] >= 1
    # 500 raw units = $0.05 = 5 ticks
    assert out["mean_spread_ticks"] == pytest.approx(5.0)


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        spread_dynamics_bench(tmp_path)
