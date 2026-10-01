"""Tests for manova — one-way omnibus statistics."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.manova import bench_manova, manova


def test_shifted_groups_reject():
    rng = np.random.default_rng(0)
    groups = [rng.standard_normal((50, 3)) + s for s in (0.0, 1.5, 2.5)]
    out = manova(groups)
    assert out["wilks_lambda"] < 0.6
    assert out["wilks_p"] < 0.001


def test_null_groups_respect_size():
    rng = np.random.default_rng(2)
    groups = [rng.standard_normal((50, 3)) for _ in range(3)]
    out = manova(groups)
    assert out["wilks_p"] > 0.01


def test_statistic_ordering():
    rng = np.random.default_rng(1)
    groups = [rng.standard_normal((40, 3)) + s * 3.0 for s in (0.0, 1.0)]
    out = manova(groups)
    assert out["roy_max"] <= out["hotelling_lawley"] + 1e-9
    assert 0 < out["wilks_lambda"] <= 1


def test_fail_closed_single_group():
    with pytest.raises(ValueError):
        manova([np.ones((10, 2))])


def test_fail_closed_bad_shape():
    with pytest.raises(ValueError):
        manova([np.ones((10, 2)), np.ones((10, 3))])


def test_bench():
    out = bench_manova()
    assert out["score"] == 1.0
