"""Tests for median_tests — Mood's median test."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.median_tests import (
    bench_median_tests,
    mood_median,
    pairwise_median_contrast,
)


def test_shift_rejected():
    rng = np.random.default_rng(0)
    out = mood_median(
        rng.standard_normal(50), rng.standard_normal(50), rng.standard_normal(50) + 1.5
    )
    assert out["p"] < 0.01


def test_null_not_rejected():
    rng = np.random.default_rng(1)
    out = mood_median(rng.standard_normal(50), rng.standard_normal(50), rng.standard_normal(50))
    assert out["p"] > 0.005


def test_pairwise_contrast():
    rng = np.random.default_rng(2)
    out = pairwise_median_contrast(rng.standard_normal(40), rng.standard_normal(40) + 1.4)
    assert out["p_two_sample"] < 0.01


def test_fail_closed_one_group():
    with pytest.raises(ValueError):
        mood_median(np.arange(20.0))


def test_bench():
    out = bench_median_tests()
    assert out["score"] == 1.0
