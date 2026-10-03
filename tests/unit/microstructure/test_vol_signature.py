from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.vol_signature import (
    _rv_at_tau,
    _signature,
    lobster_signature,
    signature_bench,
    sim_signature,
)


def test_rv_at_tau_flat_diffusive() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(0.0, 100.0, 0.05)
    m = 100 + np.cumsum(rng.normal(0, 0.05, t.size))
    rv_fine = _rv_at_tau(t, m, 0.5)
    rv_coarse = _rv_at_tau(t, m, 10.0)
    assert rv_fine is not None and rv_coarse is not None
    # diffusive noise: per-second RV roughly constant across τ
    assert 0.3 < rv_fine / rv_coarse < 3.0


def test_rv_too_short_returns_none() -> None:
    t = np.array([0.0, 1.0])
    m = np.array([100.0, 101.0])
    assert _rv_at_tau(t, m, 5.0) is None


def test_signature_shape() -> None:
    t = list(np.arange(0.0, 60.0, 0.5))
    m = list(100 + np.cumsum(np.random.default_rng(1).normal(0, 0.1, len(t))))
    out = _signature(t, m)
    assert out["n_samples"] == len(t)
    assert "0.5" in out["rv_per_tau"]


def test_lobster_signature_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        for i in range(50):
            wm.writerow([34200.0 + i, 1, i + 1, 100, 2500, 1])
            ask = 4000 + (i % 3) * 100
            wo.writerow([ask, 100, 3000, 100])
    out = lobster_signature(tmp_path)
    assert out["n_samples"] == 50


def test_sim_signature_runs() -> None:
    out = sim_signature(horizon=3000, seed=3)
    assert out["n_samples"] > 500


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        signature_bench(tmp_path)
