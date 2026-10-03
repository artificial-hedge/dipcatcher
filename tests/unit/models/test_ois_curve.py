"""Tests for ois_curve — OIS zero-curve bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.ois_curve import (
    bench_ois_curve,
    bootstrap_zero_curve,
    discount_factor,
    forward_rate,
    par_swap_rate,
)

DEP_T = np.array([0.25, 0.5])
DEP_R = np.array([0.03, 0.032])
SWAP_T = np.array([1.0, 2.0, 3.0, 5.0])
SWAP_R = np.array([0.034, 0.037, 0.040, 0.045])


def test_repricing_exact():
    c = bootstrap_zero_curve(DEP_T, DEP_R, SWAP_T, SWAP_R)
    for t, r in zip(SWAP_T, SWAP_R, strict=True):
        assert par_swap_rate(c, float(t)) == pytest.approx(float(r), abs=1e-10)


def test_discounts_decrease():
    c = bootstrap_zero_curve(DEP_T, DEP_R, SWAP_T, SWAP_R)
    dd = np.asarray(c["discount_factors"])
    assert np.all(np.diff(dd) < 0)
    assert np.all(dd < 1.0)


def test_upward_sloping_zero():
    c = bootstrap_zero_curve(DEP_T, DEP_R, SWAP_T, SWAP_R)
    zz = np.asarray(c["zero_rates"])
    assert np.all(np.diff(zz) >= -1e-12)


def test_forward_positive():
    c = bootstrap_zero_curve(DEP_T, DEP_R, SWAP_T, SWAP_R)
    f = forward_rate(c, 3.0, 5.0)
    assert f > max(SWAP_R)


def test_discount_interp():
    c = bootstrap_zero_curve(DEP_T, DEP_R, SWAP_T, SWAP_R)
    d3 = discount_factor(c, 3.0)
    d35 = discount_factor(c, 3.5)
    d4 = discount_factor(c, 4.0)
    assert d3 > d35 > d4 > discount_factor(c, 5.0)


def test_bad_inputs():
    with pytest.raises(ValueError):
        bootstrap_zero_curve(np.array([]), np.array([]), SWAP_T, SWAP_R)
    with pytest.raises(ValueError):
        bootstrap_zero_curve(DEP_T, DEP_R, np.array([2.0, 1.0]), np.array([0.03, 0.04]))


def test_bench():
    assert bench_ois_curve()["score"] == 1.0
