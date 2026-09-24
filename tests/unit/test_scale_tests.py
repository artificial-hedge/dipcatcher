"""Tests for metrics/scale_tests.py — equality-of-variance tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.metrics.scale_tests import (
    bartlett_test,
    brown_forsythe,
    fligner_killeen,
    levene_test,
)


def test_equal_variances_not_rejected() -> None:
    rng = np.random.default_rng(0)
    g1 = rng.standard_normal(300)
    g2 = rng.standard_normal(300)
    g3 = rng.standard_normal(300)
    for fn in (bartlett_test, levene_test, brown_forsythe, fligner_killeen):
        assert fn(g1, g2, g3)["pvalue"] > 0.05


def test_unequal_variances_rejected() -> None:
    rng = np.random.default_rng(1)
    g1 = rng.standard_normal(300)
    g2 = 4.0 * rng.standard_normal(300)
    for fn in (bartlett_test, levene_test, brown_forsythe, fligner_killeen):
        assert fn(g1, g2)["pvalue"] < 0.01


def test_brown_forsythe_robust_to_heavy_tails() -> None:
    rng = np.random.default_rng(2)
    g1 = rng.standard_t(3, size=400)
    g2 = rng.standard_t(3, size=400)  # same scale, heavy tails
    assert brown_forsythe(g1, g2)["pvalue"] > 0.05


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        bartlett_test(np.array([1.0, 2.0]))  # one group
    with pytest.raises(ValueError):
        levene_test(np.array([1.0]), np.array([2.0, 3.0]))  # group too small
