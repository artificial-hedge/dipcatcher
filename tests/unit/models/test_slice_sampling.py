"""Tests for slice_sampling."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.slice_sampling import bench_slice_sampling, slice_sample


def test_gamma_kernel_samples():
    def lg(x):
        return float(2 * np.log(max(x, 1e-300)) - x) if x > 0 else -np.inf

    d = slice_sample(lg, 1.0, 1500, w=1.5, seed=0)
    assert abs(d.mean() - 3.0) < 0.5


def test_deterministic_given_seed():
    def lg(x):
        return float(-0.5 * x * x)

    a = slice_sample(lg, 0.0, 50, seed=7)
    b = slice_sample(lg, 0.0, 50, seed=7)
    assert np.allclose(a, b)


def test_fail_closed_bad_logf():
    with pytest.raises(ValueError):
        slice_sample(lambda x: float("nan"), 0.0, 10)


def test_bench():
    out = bench_slice_sampling()
    assert out["score"] == 1.0
