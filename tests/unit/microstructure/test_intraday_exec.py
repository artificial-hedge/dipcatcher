"""Tests for microstructure/intraday_exec.py."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.microstructure.intraday_exec import (
    _bucket_stats,
    intraday_exec_bench,
    sim_intraday_exec,
)


def _flat_mids(n=2000):
    rng = np.random.default_rng(0)
    m = 1000.0 + np.cumsum(rng.normal(0, 0.5, n))
    return np.arange(n, dtype=float), m


def test_bucket_stats_partition():
    times, mids = _flat_mids()
    execs = [(float(t), 1, mids[t] + 1.0, 100) for t in range(10, 1990, 20)]
    buckets = _bucket_stats(execs, times, mids, n_bins=5)
    assert {b["bin"] for b in buckets} == {0, 1, 2, 3, 4}
    assert sum(b["n"] for b in buckets) <= len(execs)
    for b in buckets:
        assert np.isfinite(b["effective_mean"])
        assert 0.0 <= b["buy_share"] <= 1.0


def test_effective_positive_for_marketable_buy():
    times, mids = _flat_mids()
    execs = [(float(t), 1, mids[t] + 2.0, 10) for t in range(100, 1900, 50)]
    buckets = _bucket_stats(execs, times, mids, n_bins=3)
    assert all(b["effective_mean"] > 0 for b in buckets)


def test_last_event_closed_bin():
    times, mids = _flat_mids()
    execs = [(1999.0, 1, mids[1999], 5)]
    buckets = _bucket_stats(execs, times, mids, n_bins=4)
    # single event: bins exist only for filled buckets
    assert all(b["n"] >= 1 for b in buckets)


def test_empty_execs():
    times, mids = _flat_mids()
    assert _bucket_stats([], times, mids) == []


def test_sim_runs_and_flattish():
    out = sim_intraday_exec(None, seed=0, horizon=20000)
    assert out["n_execs"] > 50
    effs = [b["effective_mean"] for b in out["buckets"]]
    assert len(effs) >= 5
    # sim has no intraday clock — bucket spread should be small vs range
    assert max(effs) - min(effs) < 5.0


def test_bench_missing_tape(tmp_path):
    with pytest.raises(FileNotFoundError):
        intraday_exec_bench(tmp_path)
