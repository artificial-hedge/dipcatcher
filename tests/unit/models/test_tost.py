"""Tests for tost — equivalence testing."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.tost import (
    bench_tost,
    tost_correlation,
    tost_means,
    tost_ratio,
)


def test_equivalent_pair():
    rng = np.random.default_rng(0)
    a = rng.normal(size=300)
    b = rng.normal(loc=0.03, size=300)
    out = tost_means(a, b, theta=0.4)
    assert out["equivalent"] == 1.0
    assert out["p_tost"] < 0.05


def test_non_equivalent_pair():
    rng = np.random.default_rng(1)
    a = rng.normal(size=300)
    b = rng.normal(loc=1.2, size=300)
    out = tost_means(a, b, theta=0.4)
    assert out["equivalent"] == 0.0
    assert out["p_tost"] > 0.2


def test_paired_variant():
    rng = np.random.default_rng(2)
    a = rng.normal(size=200)
    b = a + rng.normal(scale=0.1, size=200)
    out = tost_means(a, b, theta=0.5, paired=True)
    assert out["equivalent"] == 1.0


def test_one_sample():
    rng = np.random.default_rng(3)
    a = rng.normal(loc=0.02, size=250)
    out = tost_means(a, theta=0.3)
    assert out["equivalent"] == 1.0


def test_correlation_variant():
    out = tost_correlation(0.01, 500, theta=0.25)
    assert out["equivalent"] == 1.0
    out2 = tost_correlation(0.5, 500, theta=0.25)
    assert out2["equivalent"] == 0.0


def test_ratio_variant():
    rng = np.random.default_rng(4)
    a = rng.lognormal(0.0, 0.2, 200)
    b = rng.lognormal(0.01, 0.2, 200)
    out = tost_ratio(a, b, theta=0.3)
    assert out["p_tost"] < 0.5


def test_fail_closed_short():
    with pytest.raises(ValueError):
        tost_means(np.ones(2))


def test_fail_closed_bad_corr():
    with pytest.raises(ValueError):
        tost_correlation(1.5, 100)


def test_bench():
    out = bench_tost()
    assert out["score"] == 1.0
