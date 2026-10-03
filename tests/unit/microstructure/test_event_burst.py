from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.event_burst import (
    _burstiness,
    burstiness_bench,
    lobster_burstiness,
    sim_burstiness,
)


def test_burstiness_poisson() -> None:
    rng = np.random.default_rng(0)
    gaps = rng.exponential(1.0, 5000)
    times = np.cumsum(gaps)
    out = _burstiness(times)
    assert abs(out["burstiness_B"]) < 0.1  # ~0 for Poisson


def test_burstiness_bursty() -> None:
    rng = np.random.default_rng(0)
    # clusters: 10 events 0.001 apart, then a 1.0 gap
    times = np.concatenate([np.cumsum(rng.uniform(0, 0.001, 10)) + k for k in range(500)])
    out = _burstiness(times)
    assert out["burstiness_B"] > 0.7  # B≈0.80 for this construction


def test_too_few() -> None:
    assert _burstiness(np.array([0.0, 1.0]))["ok"] is False


def test_lobster_burstiness_csv(tmp_path: Path) -> None:
    msg = tmp_path / "AMZN_2012-06-21_34200000_57600000_message_10.csv"
    ob = tmp_path / "AMZN_2012-06-21_34200000_57600000_orderbook_10.csv"
    with msg.open("w", newline="") as fm, ob.open("w", newline="") as fo:
        wm, wo = csv.writer(fm), csv.writer(fo)
        t = 34200.0
        for i in range(150):
            # bursty: bursts of 5 then a gap
            t += 0.001 if i % 5 else 0.5
            wm.writerow([t, 1, i + 1, 10, 4000, 1])
            wo.writerow([4000, 100, 3000, 100])
    out = lobster_burstiness(tmp_path)
    assert out["all"]["burstiness_B"] > 0.3
    assert "submission" in out


def test_sim_burstiness_runs() -> None:
    out = sim_burstiness(horizon=2000, seed=3)
    assert "burstiness_B" in out["all"]


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        burstiness_bench(tmp_path)
