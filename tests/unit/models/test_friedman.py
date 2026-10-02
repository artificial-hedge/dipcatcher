"""Tests for friedman — blocked rank tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.friedman import bench_friedman, friedman_test, page_l


def test_ordered_effect_rejected():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((20, 1))
    x = base + np.array([0.0, 0.5, 1.0, 1.5])[None, :] + 0.3 * rng.standard_normal((20, 4))
    out = friedman_test(x)
    assert out["p"] < 0.01
    assert out["kendall_w"] > 0.3


def test_null_not_rejected():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((20, 4))
    out = friedman_test(x)
    assert out["p"] > 0.005


def test_page_l_ordered():
    rng = np.random.default_rng(2)
    x = np.tile(np.arange(4)[None, :] * 0.8, (20, 1)) + 0.3 * rng.standard_normal((20, 4))
    out = page_l(x)
    assert out["p"] < 0.01
    assert out["z"] > 2.0


def test_fail_closed_bad_shape():
    with pytest.raises(ValueError):
        friedman_test(np.ones((10, 2)))


def test_bench():
    out = bench_friedman()
    assert out["score"] == 1.0
