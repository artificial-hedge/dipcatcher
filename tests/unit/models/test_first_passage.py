"""Unit tests for quant_fund.models.first_passage."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.first_passage import (
    bench_first_passage,
    fp_cdf,
    fp_mean,
    siegmund_lift,
)


def test_cdf_increasing_and_bounded() -> None:
    t = np.linspace(0.1, 5.0, 20)
    c = fp_cdf(t, 0.0, 0.8, 0.5, 1.0)
    assert np.all(c >= 0.0) and np.all(c <= 1.0)
    assert np.all(np.diff(c) > 0)


def test_mean_hitting_time() -> None:
    assert fp_mean(0.0, 0.8, 0.5, 1.0) == pytest.approx(1.0 / 0.8)


def test_siegmund_lift_positive_scaling() -> None:
    l1 = siegmund_lift(0.5, 0.1)
    l2 = siegmund_lift(0.5, 0.4)
    assert l1 > 0
    assert l2 == pytest.approx(2.0 * l1)


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        fp_cdf(np.array([1.0]), 0.0, -0.8, 0.5, 1.0)
    with pytest.raises(ValueError):
        fp_cdf(np.array([-1.0]), 0.0, 0.8, 0.5, 1.0)
    with pytest.raises(ValueError):
        fp_cdf(np.array([1.0]), 0.0, 0.8, 0.5, 0.0)
    with pytest.raises(ValueError):
        siegmund_lift(0.0, 0.1)


def test_bench_contract() -> None:
    out = bench_first_passage()
    assert out["score"] == 1.0
    assert out["synthetic_fp_max_err"] < 0.05
