"""Tests for e_divisive — energy changepoints."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.e_divisive import bench_e_divisive, e_divisive


def _two_shift(seed: int = 0):
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.normal(0.0, 0.4, 100), rng.normal(1.5, 0.4, 100), rng.normal(-0.5, 0.4, 100)]
    )


def test_detects_two_shifts():
    y = _two_shift()
    out = e_divisive(y, r=49, seed=0)
    cps = np.asarray(out["changepoints"])
    assert cps.size >= 2
    assert any(abs(c - 100) <= 12 for c in cps)
    assert any(abs(c - 200) <= 12 for c in cps)


def test_no_false_positives_on_iid():
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    out = e_divisive(y, r=49, seed=1, min_seg=15)
    cps = np.asarray(out["changepoints"])
    assert cps.size <= 1  # mostly zero on iid


def test_multivariate_input():
    rng = np.random.default_rng(2)
    x = np.vstack([rng.normal(size=(120, 2)), rng.normal(loc=2.0, size=(120, 2))])
    out = e_divisive(x, r=49, seed=2)
    cps = np.asarray(out["changepoints"])
    assert cps.size >= 1
    assert any(abs(c - 120) <= 20 for c in cps)


def test_deterministic():
    y = _two_shift(seed=3)
    a = e_divisive(y, r=29, seed=5)
    b = e_divisive(y, r=29, seed=5)
    assert np.allclose(a["changepoints"], b["changepoints"])


def test_fail_closed_short():
    with pytest.raises(ValueError):
        e_divisive(np.ones(15))


def test_fail_closed_constant():
    with pytest.raises(ValueError):
        e_divisive(np.ones(100))


def test_bench():
    out = bench_e_divisive()
    assert out["score"] == 1.0
