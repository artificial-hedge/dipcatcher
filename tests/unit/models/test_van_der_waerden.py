"""Tests for van_der_waerden — normal-scores k-sample test."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.van_der_waerden import bench_van_der_waerden, van_der_waerden


def test_shift_rejected():
    rng = np.random.default_rng(0)
    out = van_der_waerden(
        rng.standard_normal(50), rng.standard_normal(50), rng.standard_normal(50) + 1.3
    )
    assert out["p"] < 0.01


def test_null_not_rejected():
    rng = np.random.default_rng(1)
    out = van_der_waerden(rng.standard_normal(50), rng.standard_normal(50))
    assert out["p"] > 0.005


def test_two_groups():
    rng = np.random.default_rng(2)
    out = van_der_waerden(rng.standard_normal(60) + 1.0, rng.standard_normal(60))
    assert out["p"] < 0.01


def test_fail_closed_one_group():
    with pytest.raises(ValueError):
        van_der_waerden(np.arange(20.0))


def test_bench():
    out = bench_van_der_waerden()
    assert out["score"] == 1.0
