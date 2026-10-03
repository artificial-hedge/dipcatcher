from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.streak_stats import (
    _run_lengths,
    _streak_stats,
    lobster_streaks,
    sim_streaks,
    streak_bench,
)


def test_run_lengths() -> None:
    runs = _run_lengths([1, 1, 1, -1, -1, 1])
    assert runs.tolist() == [3.0, 2.0, 1.0]


def test_iid_signs_near_geometric() -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    signs = rng.choice([-1, 1], size=20000).tolist()
    out = _streak_stats(_run_lengths(signs))
    assert abs(out["excess_mass_gt5_vs_geo"]) < 0.02


def test_correlated_signs_fat_tail() -> None:
    signs = ([1] * 20 + [-1] * 20) * 25
    out = _streak_stats(_run_lengths(signs))
    assert out["excess_mass_gt5_vs_geo"] > 0.02


def test_lobster_streaks_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    with msg.open("w", newline="") as f:
        w = csv.writer(f)
        for i in range(3):
            w.writerow([34200.0 + i, 4, i + 1, 50, 5000, -1])  # buy-initiated
        w.writerow([34203.5, 4, 9, 50, 5000, 1])  # sell-initiated
    out = lobster_streaks(tmp_path)
    assert out["n_execs"] == 4
    assert out["mean_run"] == pytest.approx(2.0)


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        streak_bench(tmp_path)


def test_sim_streaks_runs() -> None:
    out = sim_streaks(horizon=1500, seed=3)
    assert out["n_runs"] > 10
