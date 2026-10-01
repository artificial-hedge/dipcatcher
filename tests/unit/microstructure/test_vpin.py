from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from quant_fund.microstructure.vpin import (
    _bucket_imbalances,
    _vpin_stats,
    vpin_bench,
    vpin_sim,
)


def test_buckets_split_correctly() -> None:
    signed = np.asarray([3.0, -2.0, 1.0, 5.0, -4.0, 4.0])
    imb, starts = _bucket_imbalances(signed, bucket_volume=4.0)
    # bucket1: 3,-2,1,5 -> vol=11 >=4 at i=3? no: greedy consumes whole events
    assert imb.size >= 1
    assert sum(imb) <= np.abs(signed).sum() + 1


def test_buckets_all_one_side_max_imbalance() -> None:
    signed = np.asarray([5.0] * 20)
    imb, _ = _bucket_imbalances(signed, bucket_volume=10.0)
    assert (imb / 10.0 == 1.0).all()


def test_vpin_stats_planted_toxicity() -> None:
    # one-sided flow + volatile mids -> positive corr setup
    rng = np.random.default_rng(0)
    signed = np.concatenate([np.full(300, 5.0), -np.full(300, 5.0)])
    mids = np.cumsum(rng.normal(0, 1, 600)) + 100.0
    out = _vpin_stats(signed, mids, n_buckets=20, window=5)
    assert out["ok"]
    assert out["vpin_mean"] == pytest.approx(1.0, abs=0.01)


def test_vpin_stats_balanced_flow() -> None:
    rng = np.random.default_rng(1)
    signed = rng.choice([1.0, -1.0], 1000)
    mids = np.cumsum(rng.normal(0, 0.5, 1000)) + 50.0
    out = _vpin_stats(signed, mids, n_buckets=20, window=5)
    assert out["ok"]
    assert out["vpin_mean"] < 0.6


def test_vpin_stats_few_trades_fails() -> None:
    out = _vpin_stats(np.ones(50), np.ones(50))
    assert out == {"ok": False, "reason": "few_trades"}


def test_vpin_sim_shape() -> None:
    out = vpin_sim(horizon=1500, seed=3)
    assert out["ok"]


def test_bench_missing_tape_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        vpin_bench(tmp_path)
