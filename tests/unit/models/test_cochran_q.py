"""Tests for cochran_q — Cochran's Q."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.cochran_q import bench_cochran_q, cochran_q


def test_column_effect_rejected():
    rng = np.random.default_rng(0)
    probs = np.array([0.15, 0.4, 0.6, 0.8])
    x = (rng.random((80, 4)) < probs[None, :]).astype(float)
    assert cochran_q(x)["p"] < 0.01


def test_iid_not_rejected():
    rng = np.random.default_rng(1)
    x = (rng.random((80, 4)) < 0.5).astype(float)
    assert cochran_q(x)["p"] > 0.005


def test_informative_blocks_counted():
    rng = np.random.default_rng(2)
    x = (rng.random((50, 3)) < np.array([0.2, 0.5, 0.8])[None, :]).astype(float)
    out = cochran_q(x)
    assert 0 < out["informative_blocks"] <= 50


def test_fail_closed_nonbinary():
    with pytest.raises(ValueError):
        cochran_q(np.array([[0.0, 0.5, 1.0]] * 10))


def test_bench():
    out = bench_cochran_q()
    assert out["score"] == 1.0
