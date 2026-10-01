"""Tests for cancel_lead: informed-withdrawal drift."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from quant_fund.microstructure.cancel_lead import (
    _lead_stats,
    cancel_lead_bench,
    lobster_cancel_lead,
    sim_cancel_lead,
)


def test_lead_stats_drift() -> None:
    rng = np.random.default_rng(0)
    n = 300
    mid_times = np.arange(n, dtype=float)
    # mid drops after a buy-side cancel (sign +1 -> adverse drift -1)
    mids = 100.0 + np.cumsum(rng.normal(0, 0.1, n)) - np.arange(n) * 0.01
    canc_t = np.arange(10, n - 10, dtype=float)
    signs = np.ones(len(canc_t))
    out = _lead_stats(signs, canc_t, mid_times, mids)
    assert out["ok"]
    assert out["per_horizon"]["5.0s"]["mean_signed_drift_ticks"] < 0
    assert out["per_horizon"]["5.0s"]["share_adverse"] > 0.5


def test_lead_stats_null() -> None:
    rng = np.random.default_rng(1)
    n = 300
    mid_times = np.arange(n, dtype=float)
    mids = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    canc_t = np.arange(10, n - 10, dtype=float)
    signs = np.ones(len(canc_t))
    out = _lead_stats(signs, canc_t, mid_times, mids)
    assert abs(out["per_horizon"]["5.0s"]["mean_signed_drift_ticks"]) < 0.1


def test_lobster_csv(tmp_path) -> None:
    msg = tmp_path / "m.csv"
    ob = tmp_path / "o.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        t = 34200.0
        for i in range(1, 80):
            t += 0.01
            wm.writerow([t, 1, i, 10, 4000, 1])  # submit at bid
            wo.writerow([4100, 500, 4000, 500])
            t += 0.01
            wm.writerow([t, 3, i, 0, 0, 1])  # delete at touch
            # mid drifts down on each delete row (ask 4050-i stays > bid)
            wo.writerow([4050 - i, 500, 3950, 500])
    out = lobster_cancel_lead(msg, ob)
    assert out["ok"]
    assert out["per_horizon"]["0.1s"]["mean_signed_drift_ticks"] < 0


def test_sim_runs() -> None:
    out = sim_cancel_lead(horizon=8000, seed=3)
    assert out["ok"]


def test_missing_tape(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        cancel_lead_bench(tmp_path)
