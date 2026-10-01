from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.lob_resilience import (
    _resilience_stats,
    lobster_resilience,
    resilience_bench,
    sim_resilience,
)


def test_resilience_math() -> None:
    # sizes: 100 -> deplete to 10 -> recover to 60 at t=1.0
    times = [0.0, 0.5, 1.0]
    sizes = [100, 10, 60]
    out = _resilience_stats(times, sizes)
    assert out["n_depletions"] == 1
    assert out["n_recovered"] == 1
    assert out["median_refill_s"] == pytest.approx(0.5)


def test_unrecovered_counted() -> None:
    times = [0.0, 0.5, 1.0]
    sizes = [100, 10, 12]  # never recovers to 50
    out = _resilience_stats(times, sizes)
    assert out["n_depletions"] == 1
    assert out["n_unrecovered"] == 1
    assert out["n_recovered"] == 0


def test_no_depletion() -> None:
    out = _resilience_stats([0.0, 1.0], [100, 90])
    assert out["n_depletions"] == 0
    assert out["recovery_share"] == 0.0


def test_lobster_resilience_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        wm.writerow([34200.0, 1, 1, 100, 2500, 1])
        wo.writerow([4000, 100, 3000, 100])
        wm.writerow([34200.5, 1, 2, 100, 2500, 1])
        wo.writerow([4000, 15, 3000, 100])  # ask size depleted 100->15
        wm.writerow([34201.0, 1, 3, 100, 2500, 1])
        wo.writerow([4000, 60, 3000, 100])  # recovered to 60 >= 50
    out = lobster_resilience(tmp_path, side="ask")
    assert out["n_depletions"] == 1
    assert out["n_recovered"] == 1
    assert out["median_refill_s"] == pytest.approx(0.5)


def test_sim_resilience_runs() -> None:
    out = sim_resilience(horizon=1500, seed=3)
    assert "n_depletions" in out


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resilience_bench(tmp_path)
