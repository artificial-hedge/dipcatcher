from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.quote_place import (
    _distance_hist,
    lobster_placement,
    quote_place_bench,
    sim_placement,
)


def test_hist_partitions() -> None:
    d = np.asarray([-10.0, -3.0, -0.5, 0.0, 1.0, 2.5, 4.0, 6.0, 10.0, 40.0, 100.0])
    h = _distance_hist(d)
    assert sum(h.values()) == 11
    assert h["-1..0"] == 2 and h["0..1"] == 1 and h["gt55"] == 1


def test_lobster_placement_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    # book rows: 1 level each [ap, asz, bp, bsz]
    rows_ob = [
        [5010, 100, 4990, 100],
        [5010, 100, 4990, 100],
        [5010, 100, 4990, 100],
        [5010, 100, 4990, 200],
        [5010, 100, 4990, 200],
    ]
    rows_msg = [
        [34200.0, 1, 1, 100, 5010, -1],  # seed
        [34201.0, 1, 2, 100, 4990, 1],  # buy at touch -> dist 0
        [34202.0, 1, 3, 50, 4800, 1],  # buy deep -> dist +19 ticks... (4990-4800)/100=1.9
        [34203.0, 1, 4, 100, 4995, 1],  # buy inside spread -> -0.5 ticks
        [34204.0, 1, 5, 100, 5100, -1],  # sell deeper -> (5100-5010)/100=0.9
    ]
    with msg.open("w", newline="") as f:
        for r in rows_msg:
            csv.writer(f).writerow(r)
    with ob.open("w", newline="") as f:
        for r in rows_ob:
            csv.writer(f).writerow([str(x) for x in r])
    out = lobster_placement(tmp_path)
    assert out["n_submissions"] == 4
    assert out["share_at_touch"] == pytest.approx(0.25)
    assert out["share_improves_spread"] == pytest.approx(0.25)


def test_sim_placement_runs() -> None:
    out = sim_placement(horizon=1500, seed=3)
    assert out["n_submissions"] > 50
    assert out["share_behind_touch"] is not None


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        quote_place_bench(tmp_path)
