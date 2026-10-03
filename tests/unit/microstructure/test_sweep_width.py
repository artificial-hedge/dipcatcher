"""Tests for sweep_width: aggressor impulse footprint."""

from __future__ import annotations

import csv

import pytest

from quant_fund.microstructure.sweep_width import (
    _footprint,
    lobster_sweep_width,
    sim_sweep_width,
    sweep_width_bench,
)


def test_footprint_widths() -> None:
    # one 3-level walk + one single print
    clusters = [
        (3, [(100.0, 10.0), (101.0, 10.0), (102.0, 5.0)]),
        (1, [(99.0, 20.0)]),
    ]
    out = _footprint(clusters)
    assert out["n_clusters"] == 2
    assert out["width_ge2_share"] == 0.5
    assert out["max_width"] == 3
    assert out["mean_walk_ticks"] == 1.0


def test_lobster_csv(tmp_path) -> None:
    msg = tmp_path / "m.csv"
    with msg.open("w", newline="") as fm:
        w = csv.writer(fm)
        # one aggressor sweeps three resting orders at two prices, same t
        for oid, px in [(1, 4100), (2, 4100), (3, 4200)]:
            w.writerow([34200.0, 1, oid, 10, px, -1])
            w.writerow([34200.0, 4, oid, 10, px, -1])
        # a lone width-1 exec at a different timestamp
        w.writerow([34201.0, 1, 9, 10, 4000, 1])
        w.writerow([34201.0, 4, 9, 10, 4000, 1])
    out = lobster_sweep_width(msg)
    assert out["n_clusters"] == 2
    assert out["width_hist"] == {"1": 1, "2": 1}
    assert out["max_width"] == 2


def test_sim_runs() -> None:
    out = sim_sweep_width(horizon=8000, seed=3)
    assert out["ok"]
    assert out["max_width"] == 1  # unit-size aggressors


def test_missing_tape(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        sweep_width_bench(tmp_path)
