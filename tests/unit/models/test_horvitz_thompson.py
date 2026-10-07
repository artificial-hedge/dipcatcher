"""Tests for horvitz_thompson — design-based estimation."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.horvitz_thompson import (
    bench_horvitz_thompson,
    hajek_mean,
    horvitz_thompson,
)


def test_equal_pi_is_srs():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(50)
    pi = np.full(50, 0.1)
    out = horvitz_thompson(y, pi)
    assert out["total"] == pytest.approx(y.sum() * 10)


def test_pps_unbiased_mean():
    rng = np.random.default_rng(1)
    y_pop = rng.gamma(2.0, 10.0, 3000)
    pi = np.clip(300 * y_pop / y_pop.sum(), 1e-3, 0.5)
    sel = rng.random(3000) < pi
    out = hajek_mean(y_pop[sel], pi[sel])
    assert abs(out["mean"] - y_pop.mean()) / y_pop.mean() < 0.15


def test_se_positive():
    rng = np.random.default_rng(2)
    out = horvitz_thompson(rng.standard_normal(30), np.full(30, 0.2))
    assert out["se"] > 0


def test_fail_closed_zero_pi():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        horvitz_thompson(rng.standard_normal(10), np.zeros(10))


def test_bench():
    out = bench_horvitz_thompson()
    assert out["synthetic_score"] == 1.0
