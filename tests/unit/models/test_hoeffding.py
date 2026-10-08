"""Tests for hoeffding — Hoeffding's D."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.hoeffding import bench_hoeffding, hoeffding_d


def test_circle_detected():
    rng = np.random.default_rng(0)
    t = rng.random(100) * 2 * np.pi
    x = np.cos(t) + 0.1 * rng.standard_normal(100)
    y = np.sin(t) + 0.1 * rng.standard_normal(100)
    out = hoeffding_d(x, y, n_perm=199, seed=0)
    assert out["d"] > 0
    assert out["p"] < 0.01


def test_iid_not_rejected():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(100)
    y = rng.standard_normal(100)
    out = hoeffding_d(x, y, n_perm=199, seed=1)
    assert out["p"] > 0.005


def test_monotone_detected():
    rng = np.random.default_rng(2)
    x = rng.random(100)
    y = x + 0.1 * rng.standard_normal(100)
    out = hoeffding_d(x, y, n_perm=99, seed=2)
    assert out["nd"] > 0.3


def test_fail_closed_small():
    with pytest.raises(ValueError):
        hoeffding_d(np.arange(5.0), np.arange(5.0))


def test_bench():
    out = bench_hoeffding()
    assert out["synthetic_score"] == 1.0
