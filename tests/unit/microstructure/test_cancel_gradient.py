"""Tests for cancel_gradient: spatial cancel propensity."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from quant_fund.microstructure.cancel_gradient import (
    _gradient,
    cancel_gradient_bench,
    lobster_cancel_gradient,
    sim_cancel_gradient,
)


def test_gradient_touch_skew() -> None:
    rng = np.random.default_rng(0)
    # cancels concentrated at the touch; depth spread over 0..10 ticks
    cancel_d = np.concatenate([np.zeros(400), rng.uniform(1, 10, 100)])
    depth_d = rng.integers(0, 11, 2000).astype(float)
    depth_w = np.ones(2000)
    out = _gradient(cancel_d, depth_d, depth_w)
    assert out["ok"]
    assert out["at_touch_share"] == pytest.approx(0.8)
    touch = out["buckets"][0]
    assert touch["propensity"] is not None and touch["propensity"] > 3.0


def test_gradient_few() -> None:
    out = _gradient(np.ones(10), np.ones(10), np.ones(10))
    assert out["ok"] is False


def test_lobster_csv(tmp_path) -> None:
    msg = tmp_path / "m.csv"
    ob = tmp_path / "o.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        t = 34200.0
        for i in range(1, 150):
            t += 0.01
            wm.writerow([t, 1, i, 10, 4000 - (i % 3) * 100, 1])  # buy submit
            wo.writerow([4100, 500, 4000, 500])
            t += 0.01
            wm.writerow([t, 3, i, 0, 0, 1])  # delete it
            wo.writerow([4100, 500, 4000, 500])
    out = lobster_cancel_gradient(msg, ob)
    assert out["ok"]
    # deletes at 4000/4100/4200 relative to best bid 4000 -> d 0/1/2 ticks
    assert out["median_distance_ticks"] <= 2.0


def test_sim_runs() -> None:
    out = sim_cancel_gradient(horizon=8000, seed=3)
    assert out["ok"]
    assert 0.0 <= out["at_touch_share"] <= 1.0


def test_missing_tape(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        cancel_gradient_bench(tmp_path)
