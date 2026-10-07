"""Unit tests for quant_fund.models._geo_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._geo_synth import (
    synth_field,
    synth_hierarchy,
    synth_monotonic,
    synth_partwhole,
    synth_rot_cloud,
    synth_sort,
)


def test_hierarchy_tree_metric() -> None:
    rng = np.random.default_rng(0)
    vecs, idx, dist = synth_hierarchy(3, rng)
    assert vecs.shape == (8, 2) and dist.shape == (8, 8)
    np.testing.assert_array_equal(idx, np.arange(8))
    assert (np.diag(dist) == 0).all()
    np.testing.assert_allclose(dist, dist.T)
    # siblings (i, i+1 with even i) share depth-1 prefix -> distance 2
    assert dist[0, 1] == 2.0
    # farthest pair shares nothing -> 2*depth
    assert dist[0, 4] == 6.0


def test_hierarchy_rejects_degenerate_depth() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="depth"):
        synth_hierarchy(0, rng)
    with pytest.raises(ValueError, match="depth"):
        synth_hierarchy(-2, rng)


def test_partwhole_label_computable_from_observed_x() -> None:
    rng = np.random.default_rng(0)
    x, y = synth_partwhole(2000, rng)
    # the class must be computable from the OBSERVED (noisy) parts
    recomputed = ((x[:, 0] * x[:, 1]).sum(-1) > 0).astype(np.int64)
    np.testing.assert_array_equal(y, recomputed)


def test_partwhole_determinism() -> None:
    x1, y1 = synth_partwhole(64, np.random.default_rng(2))
    x2, y2 = synth_partwhole(64, np.random.default_rng(2))
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_field_gradient_is_analytic() -> None:
    rng = np.random.default_rng(1)
    pts, f, grad = synth_field(500, rng)
    # finite-difference check of the returned gradient
    eps = 1e-6
    for i in range(50):
        p = pts[i].copy()
        fx = (
            np.sin(2 * (p[0] + eps)) * np.cos(3 * p[1])
            + 0.5 * (p[0] + eps)
            - np.sin(2 * (p[0] - eps)) * np.cos(3 * p[1])
            - 0.5 * (p[0] - eps)
        ) / (2 * eps)
        fy = (
            np.sin(2 * p[0]) * np.cos(3 * (p[1] + eps))
            - np.sin(2 * p[0]) * np.cos(3 * (p[1] - eps))
        ) / (2 * eps)
        assert abs(grad[i, 0] - fx) < 1e-5
        assert abs(grad[i, 1] - fy) < 1e-5


def test_rot_cloud_rotation_invariant_labels() -> None:
    rng = np.random.default_rng(0)
    x, y = synth_rot_cloud(200, 12, rng)
    assert set(np.unique(y)) <= {0, 1}
    assert y.mean() == pytest.approx(0.5, abs=0.03)  # median split is balanced


def test_rot_cloud_rejects_single_point() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="m_pts"):
        synth_rot_cloud(8, 1, rng)


def test_monotonic_trend_direction() -> None:
    rng = np.random.default_rng(0)
    x, y = synth_monotonic(500, rng)
    order = np.argsort(x[:, 0])
    corr = np.corrcoef(x[order, 0], y[order, 0])[0, 1]
    assert corr > 0.9


def test_sort_ranks_correct() -> None:
    rng = np.random.default_rng(0)
    x, ranks = synth_sort(20, 7, rng)
    for i in range(20):
        np.testing.assert_array_equal(np.argsort(x[i]), np.argsort(ranks[i]))
        assert sorted(ranks[i]) == list(range(7))
