from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_fund.microstructure.mid_jump import (
    _jump_profile,
    lobster_mid_jumps,
    mid_jump_bench,
    sim_mid_jumps,
)


def test_jump_profile_math() -> None:
    times = [0.0, 0.1, 0.2, 0.3]
    mids = [100.0, 100.0, 101.0, 101.0]  # one 1-tick move
    out = _jump_profile(times, mids)
    assert out["n_mid_moves"] == 1
    assert out["move_share"] == pytest.approx(0.3333)
    assert out["mean_abs_jump_ticks"] == pytest.approx(1.0)


def test_jump_profile_too_few() -> None:
    assert _jump_profile([0.0], [100.0])["ok"] is False


def test_lobster_mid_jumps_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        wm.writerow([34200.0, 1, 1, 100, 2500, 1])
        wo.writerow([4000, 100, 3000, 100])  # mid 3500 raw = 35 ticks
        wm.writerow([34200.5, 1, 2, 100, 2500, 1])
        wo.writerow([4000, 100, 3100, 100])  # mid 3550 → 0.5 tick move
    out = lobster_mid_jumps(tmp_path)
    assert out["n_obs"] == 1  # 2 samples -> 1 diff
    assert out["mean_abs_jump_ticks"] == pytest.approx(0.5)


def test_sim_mid_jumps_runs() -> None:
    out = sim_mid_jumps(horizon=1500, seed=3)
    assert out["n_obs"] > 100


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        mid_jump_bench(tmp_path)
