"""Tests for stale_quote: maker-age-at-fill × forward-drift pickoff lane."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from quant_fund.microstructure.stale_quote import (
    _fill_age_drift,
    lobster_stale_quote,
    sim_stale_quote,
    stale_quote_bench,
)


def test_fill_age_drift_buckets() -> None:
    rng = np.random.default_rng(0)
    ages = rng.uniform(0, 60, 300)
    drift = np.where(ages > 5.0, 2.0, 0.1)  # old makers drift adverse
    out = _fill_age_drift(ages, drift)
    assert out["ok"]
    old = [b for b in out["age_bins"] if b["age_lo_s"] >= 5.0]
    young = [b for b in out["age_bins"] if b["age_lo_s"] < 5.0]
    assert all(b["mean_signed_dmid_ticks"] == pytest.approx(2.0) for b in old)
    assert all(b["mean_signed_dmid_ticks"] == pytest.approx(0.1) for b in young)


def test_fill_age_drift_few() -> None:
    out = _fill_age_drift(np.arange(10.0), np.ones(10))
    assert out["ok"] is False


def test_lobster_csv(tmp_path) -> None:
    msg = tmp_path / "m.csv"
    ob = tmp_path / "o.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        t = 34200.0
        # resting sell submitted, then lifted by a buy aggressor
        for oid in range(1, 121):
            wm.writerow([t, 1, oid, 50, 4100, -1])  # resting sell @4100
            wo.writerow([4100, 500, 4000, 500])
            t += 0.05
            # buy aggressor lifts it: resting side -1
            wm.writerow([t, 4, oid, 50, 4100, -1])
            wo.writerow([4100 + (10 if oid % 3 == 0 else 0), 400, 4000, 500])
            t += 0.05
    out = lobster_stale_quote(msg, ob)
    assert out["ok"]
    assert out["median_age_s"] == pytest.approx(0.05)


def test_sim_ages() -> None:
    out = sim_stale_quote(horizon=8000, seed=3)
    assert out["ok"]
    assert out["median_age_s"] > 0


def test_missing_tape(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        stale_quote_bench(tmp_path)
