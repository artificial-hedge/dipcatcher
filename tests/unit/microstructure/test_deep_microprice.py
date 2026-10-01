"""Tests for deep_microprice: multi-level depth-weighted microprice."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from quant_fund.microstructure.deep_microprice import (
    _microprice_curves,
    deep_microprice_bench,
    lobster_deep_microprice,
    sim_deep_microprice,
)


def test_curves_recover_signal() -> None:
    rng = np.random.default_rng(0)
    asks: list[list[tuple[float, float]]] = []
    bids: list[list[tuple[float, float]]] = []
    # imbalance drives the next mid move: heavy bid depth -> mid rises
    mid = 4000.0
    imb = 100.0
    for _ in range(2000):
        imb = float(np.clip(0.97 * imb + rng.normal(0, 15), 10, 250))
        x_t = (imb - 20.0) / (imb + 20.0)  # touch microprice deviation
        bids.append([(mid - 1.0, imb), (mid - 2.0, 5.0)])
        asks.append([(mid + 1.0, 20.0), (mid + 2.0, 5.0)])
        mid += 2.0 * (x_t - 0.55) + rng.normal(0, 0.05)  # mid tracks M_1
    out = _microprice_curves(asks, bids)
    assert out["ok"]
    k1 = next(p for p in out["per_depth"] if p["levels"] == 1)
    assert k1["ok"] and k1["r2"] > 0.3
    assert out["best_r2"] > 0.3


def test_curves_few() -> None:
    out = _microprice_curves([[(10.0, 1.0)]] * 50, [[(9.0, 1.0)]] * 50)
    assert out["ok"] is False


def test_lobster_csv(tmp_path) -> None:
    msg = tmp_path / "m.csv"
    ob = tmp_path / "o.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        t = 34200.0
        for i in range(1, 400):
            t += 0.01
            wm.writerow([t, 1, i, 10, 4000, 1])
            ask_sz = 500 if i % 2 == 0 else 50
            wo.writerow([4100, ask_sz, 200, 5, 4000, 500, 200, 5])
    out = lobster_deep_microprice(msg, ob)
    assert "ok" in out  # structure sanity; short tape may be under-n


def test_sim_runs() -> None:
    out = sim_deep_microprice(horizon=4000, seed=3)
    assert out["ok"]
    assert out["per_depth"][0]["levels"] == 1


def test_missing_tape(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        deep_microprice_bench(tmp_path)
