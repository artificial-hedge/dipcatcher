"""event_granger contracts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.event_granger import (
    _binned_counts,
    _xcorr,
    event_granger_bench,
    lobster_event_granger,
    sim_event_granger,
)


def test_xcorr_detects_lead() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=2000)
    b = np.zeros(2000)
    b[5:] = a[:-5]  # b is a delayed by 5 bins
    xc = _xcorr(a, b, 20)
    assert int(np.argmax(np.abs(xc))) == 5 and xc[5] > 0.5


def test_xcorr_independent_zero() -> None:
    rng = np.random.default_rng(0)
    xc = _xcorr(rng.normal(size=2000), rng.normal(size=2000), 20)
    assert np.abs(xc).max() < 0.2


def test_binned_counts() -> None:
    times = np.array([0.001, 0.002, 0.015, 0.02])
    types = np.array([1, 4, 1, 3])
    c = _binned_counts(times, types, 0.01)
    assert c["submit"].sum() == 2 and c["exec"].sum() == 1 and c["delete"].sum() == 1
    assert c["submit"][0] == 1 and c["submit"][1] == 1


def test_lobster_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    rng = np.random.default_rng(0)
    with msg.open("w", newline="") as f:
        w = csv.writer(f)
        t = 34200.0
        for i in range(400):
            t += float(rng.uniform(0.002, 0.04))
            w.writerow([t, 3, i * 2 + 1, 10, 4000, 1])  # delete
            w.writerow([t + 0.002, 4, i * 2 + 2, 10, 4100, -1])  # exec 2ms later
    out = lobster_event_granger(msg)
    assert out["ok"] and out["n_events"] == 800
    assert out["delete->exec"]["peak_corr"] > 0.3


def test_sim_null_small() -> None:
    out = sim_event_granger(horizon=6000, seed=5)
    assert out["ok"]
    for pair, vals in out.items():
        if isinstance(vals, dict) and "peak_corr" in vals:
            assert abs(vals["peak_corr"]) < 0.4, pair


def test_bench_missing_tape(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        event_granger_bench(tmp_path)
