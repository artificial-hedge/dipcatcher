"""impact_instant contracts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.impact_instant import (
    _impact_curve,
    impact_instant_bench,
    lobster_impact_instant,
    sim_impact_instant,
)


def test_curve_flat_zero() -> None:
    sizes = np.tile([1.0, 10.0, 100.0], 200)
    out = _impact_curve(sizes, np.zeros(600))
    assert out["ok"] and out["alpha_loglog"] is None and out["mean_abs_dmid_ticks"] == 0.0


def test_curve_recovers_power_law() -> None:
    rng = np.random.default_rng(0)
    sizes = rng.lognormal(4.0, 1.0, 2000)
    dmid = np.round(2.0 * np.sqrt(sizes) * 0.05) * np.sign(rng.normal(size=2000) + 0.5)
    out = _impact_curve(sizes, dmid)
    assert out["ok"] and out["alpha_loglog"] is not None
    assert out["alpha_loglog"] > 0.2


def test_curve_too_few() -> None:
    out = _impact_curve(np.arange(10.0), np.zeros(10))
    assert not out["ok"]


def test_lobster_impact_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        t = 34200.0
        for oid, i in enumerate(range(120), start=1):
            sz = 10 + (i % 3) * 50
            t += 0.01
            wm.writerow([t, 4, oid, sz, 4100, -1])
            # mid jumps by sz price units after each exec
            wo.writerow([4100 + sz, 500, 4000 + sz, 500])
    out = lobster_impact_instant(msg, ob)
    # first exec has no previous mid to diff against
    assert out["ok"] and out["n"] == 119
    assert out["p_move"] > 0.6 and out["mean_abs_dmid_ticks"] > 0.5


def test_sim_unit_lots() -> None:
    out = sim_impact_instant(horizon=6000, seed=5)
    assert out["ok"]
    assert out["size_median"] == 1.0


def test_bench_missing_tape(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        impact_instant_bench(tmp_path)
