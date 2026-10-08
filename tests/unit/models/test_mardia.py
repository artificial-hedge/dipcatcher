"""Tests for mardia — multivariate normality."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.mardia import bench_mardia, mardia_test


def test_mvn_null_not_rejected():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((250, 3))
    out = mardia_test(x)
    assert out["skew_p"] > 0.005
    assert out["kurt_p"] > 0.005


def test_heavy_tail_rejected():
    rng = np.random.default_rng(1)
    x = rng.standard_t(4.0, (300, 3))
    out = mardia_test(x)
    assert out["kurt_p"] < 0.01


def test_b2p_near_expected_under_mvn():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((400, 4))
    out = mardia_test(x)
    # E[b2p] = p(p+2) = 24 for p=4
    assert abs(out["b2p"] - 24.0) < 8.0


def test_fail_closed_singular():
    with pytest.raises(ValueError):
        mardia_test(np.ones((30, 2)))


def test_bench():
    out = bench_mardia()
    assert out["synthetic_score"] == 1.0
