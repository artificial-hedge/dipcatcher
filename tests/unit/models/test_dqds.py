"""Tests for the dqds singular-value iterator."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.dqds import bench_dqds, dqds


def test_matches_svd():
    rng = np.random.RandomState(0)
    for _ in range(5):
        n = rng.randint(3, 7)
        B = np.diag(rng.rand(n) + 0.5) + np.diag(rng.rand(n - 1), 1)
        got = dqds(B)
        exp = np.sort(np.linalg.svd(B, compute_uv=False))
        assert np.linalg.norm(got - exp) < 0.05 * np.linalg.norm(exp)


def test_negative_q_fails_closed():
    """A shift larger than the smallest singular value drives q negative;
    sqrt(abs(q)) would silently fabricate 'singular values'."""
    B = np.diag([0.05, 0.5, 1.0]) + np.diag([0.1, 0.1], 1)
    with pytest.raises(ValueError):
        dqds(B, iters=60, shift=2.0)


def test_bench():
    assert bench_dqds()["synthetic_dqds_close"] == 1.0
