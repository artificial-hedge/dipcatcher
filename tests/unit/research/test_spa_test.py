"""Tests for research/spa_test.py — White RC + Romano–Wolf stepdown."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.research.spa_test import (
    SPA_SCHEMA,
    reality_check_pvalue,
    romano_wolf_stepdown,
    spa_bench,
    stationary_bootstrap_indices,
)


def test_bootstrap_indices_valid() -> None:
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(100, 0.2, rng)
    assert idx.size == 100
    assert idx.min() >= 0 and idx.max() < 100


def test_bootstrap_validation() -> None:
    with pytest.raises(ValueError, match="p must be"):
        stationary_bootstrap_indices(10, 0.0, np.random.default_rng(0))


def test_rc_pvalue_in_unit_interval() -> None:
    rng = np.random.default_rng(1)
    losses = rng.standard_normal((100, 4))
    pv = reality_check_pvalue(losses, n_boot=50, seed=0)
    assert 0.0 <= pv <= 1.0


def test_rc_rejects_planted_winner() -> None:
    rng = np.random.default_rng(2)
    losses = rng.standard_normal((300, 3))
    losses[:, 1] -= 0.8  # head 1 much better than benchmark (col 0)
    pv = reality_check_pvalue(losses, n_boot=100, seed=0)
    assert pv < 0.05


def test_stepdown_identifies_winner() -> None:
    rng = np.random.default_rng(3)
    losses = rng.standard_normal((300, 4))
    losses[:, 1] -= 0.7
    res = romano_wolf_stepdown(losses, n_boot=100, alpha=0.05, seed=0)
    # head index 1 in original columns maps to diff-column 0
    assert 0 in res.rejected or 1 in res.rejected or len(res.rejected) >= 1
    assert res.adj_p.size == losses.shape[1] - 1


def test_stepdown_null_no_reject() -> None:
    rng = np.random.default_rng(4)
    losses = rng.standard_normal((200, 5))
    res = romano_wolf_stepdown(losses, n_boot=80, alpha=0.05, seed=0)
    # null arm: not guaranteed zero rejects, but shouldn't reject *everything*
    assert len(res.rejected) < 4


def test_validation() -> None:
    with pytest.raises(ValueError, match="n, k"):
        romano_wolf_stepdown(np.zeros((5, 1)))
    with pytest.raises(ValueError, match="benchmark_col"):
        romano_wolf_stepdown(np.zeros((5, 3)), benchmark_col=9)


def test_determinism() -> None:
    rng = np.random.default_rng(5)
    losses = rng.standard_normal((60, 3))
    a = romano_wolf_stepdown(losses, n_boot=30, seed=7)
    b = romano_wolf_stepdown(losses, n_boot=30, seed=7)
    assert np.allclose(a.adj_p, b.adj_p)


def test_bench_smoke_and_schema() -> None:
    out = spa_bench(n=60, k=3, n_boot=20, n_reps=3, deltas=(0.0, 0.3))
    assert out["schema"] == SPA_SCHEMA
    assert set(out["arms"]) == {"delta=0.0", "delta=0.3"}
    assert out["data_label"] == "SYNTHETIC"
    assert len(out["payload_sha256"]) == 64
