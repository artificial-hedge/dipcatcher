"""Tests for isolation_forest — anomaly scoring."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.isolation_forest import bench_isolation_forest, isolation_forest


def test_scores_bounded():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 2))
    out = isolation_forest(x, n_trees=50, psi=64)
    s = np.asarray(out["scores"])
    assert np.all(s > 0) and np.all(s <= 1.0)


def test_outlier_scores_higher():
    rng = np.random.default_rng(1)
    core = rng.normal(size=(150, 2))
    out_pt = np.array([[8.0, -8.0]])
    x = np.vstack([core, out_pt])
    out = isolation_forest(x, n_trees=100, psi=64, seed=1)
    s = np.asarray(out["scores"])
    assert s[-1] > np.quantile(s[:-1], 0.95)


def test_extended_variant_runs():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 3))
    out = isolation_forest(x, n_trees=30, psi=64, extended=True, seed=2)
    s = np.asarray(out["scores"])
    assert s.size == 100 and np.all(np.isfinite(s))


def test_deterministic():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 2))
    a = isolation_forest(x, n_trees=30, psi=32, seed=5)
    b = isolation_forest(x, n_trees=30, psi=32, seed=5)
    assert np.allclose(a["scores"], b["scores"])


def test_fail_closed_tiny():
    with pytest.raises(ValueError):
        isolation_forest(np.ones((4, 2)))


def test_fail_closed_constant():
    with pytest.raises(ValueError):
        isolation_forest(np.ones((50, 2)))


def test_bench():
    out = bench_isolation_forest()
    assert out["synthetic_score"] == 1.0
