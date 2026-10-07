"""Tests for james_stein."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.james_stein import (
    bench_james_stein,
    empirical_bayes_shrinkage,
    james_stein,
    positive_part_js,
)


def test_js_dominates_mle():
    rng = np.random.default_rng(0)
    p = 6
    th = np.array([2.0, -1.0, 0.5, 0.0, 0.0, 1.0])
    mm = ms = 0.0
    for _ in range(200):
        x = th + rng.standard_normal(p)
        mm += ((x - th) ** 2).sum()
        ms += ((james_stein(x) - th) ** 2).sum()
    assert ms < mm


def test_positive_part_dominates_js():
    rng = np.random.default_rng(1)
    p = 6
    th = np.zeros(p)
    mj = mp = 0.0
    for _ in range(200):
        x = th + rng.standard_normal(p)
        mj += ((james_stein(x) - th) ** 2).sum()
        mp += ((positive_part_js(x) - th) ** 2).sum()
    assert mp <= mj + 1e-9


def test_eb_shapes():
    out = empirical_bayes_shrinkage(np.array([3.0, 0.0, 0.0, 0.0]), 1.0)
    assert out.shape == (4,)


def test_fail_closed_small_p():
    with pytest.raises(ValueError):
        james_stein(np.array([1.0, 2.0]), 1.0)


def test_bench():
    out = bench_james_stein()
    assert out["synthetic_score"] == 1.0
