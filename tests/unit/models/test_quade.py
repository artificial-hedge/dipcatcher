"""Tests for quade — weighted block ranks."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.quade import bench_quade, quade_test


def test_ordered_effect_rejected():
    rng = np.random.default_rng(0)
    x = np.tile(np.arange(4)[None, :] * 0.7, (20, 1)) + 0.4 * rng.standard_normal((20, 4))
    out = quade_test(x)
    assert out["p"] < 0.01


def test_null_not_rejected():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((20, 4))
    out = quade_test(x)
    assert out["p"] > 0.005


def test_t3_consistency():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((20, 4))
    out = quade_test(x)
    assert np.isfinite(out["t3"])
    assert out["t3"] >= 0.0


def test_fail_closed_shape():
    with pytest.raises(ValueError):
        quade_test(np.ones((10, 2)))


def test_bench():
    out = bench_quade()
    assert out["score"] == 1.0
