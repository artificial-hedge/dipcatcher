from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.round_lot import (
    _size_profile,
    lobster_round_lot,
    round_lot_bench,
    sim_round_lot,
)


def test_size_profile_round_share() -> None:
    sizes = np.asarray([100, 100, 200, 500, 137, 43, 1000, 77])
    out = _size_profile(sizes)
    assert out["round_lot_share"] == pytest.approx(5 / 8)
    assert out["multiple_of_100_share"] == pytest.approx(5 / 8)


def test_size_profile_empty() -> None:
    assert _size_profile(np.asarray([]))["ok"] is False


def test_lobster_round_lot_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    with msg.open("w", newline="") as f:
        w = csv.writer(f)
        for i, sz in enumerate([100, 100, 200, 73]):
            w.writerow([34200.0 + i, 4, i + 1, sz, 5000, -1])
    out = lobster_round_lot(tmp_path)
    assert out["n_trades"] == 4
    assert out["round_lot_share"] == pytest.approx(0.75)


def test_sim_round_lot_runs() -> None:
    out = sim_round_lot(horizon=1500, seed=3)
    assert out["n_trades"] > 10


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        round_lot_bench(tmp_path)
