"""Unit tests for quant_fund.models._gt_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._gt_synth import ar2_baseline, gt_data, ring_adj


def test_ring_adj_normalized() -> None:
    A = ring_adj(8)
    assert A.shape == (8, 8)
    np.testing.assert_allclose(A.sum(1), np.ones(8))
    np.testing.assert_allclose(A, A.T)
    assert (np.diag(A) == 0).all()
    # each node connects exactly its two neighbors
    assert (A[0, [1, 7]] == 0.5).all()


def test_ring_adj_rejects_too_small() -> None:
    with pytest.raises(ValueError, match="ring"):
        ring_adj(2)
    with pytest.raises(ValueError, match="ring"):
        ring_adj(1)


def test_gt_data_shape_determinism() -> None:
    X1 = gt_data(0, T=60, n=8)
    X2 = gt_data(0, T=60, n=8)
    np.testing.assert_array_equal(X1, X2)
    assert X1.shape == (60, 8)
    assert np.isfinite(X1).all()


def test_gt_data_neighbors_correlated() -> None:
    X = gt_data(0, T=300, n=8)
    corrs = [np.corrcoef(X[:, i], X[:, (i + 1) % 8])[0, 1] for i in range(8)]
    assert np.mean(corrs) > 0.5  # phase-shifted neighbors predict each other


def test_gt_data_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="T"):
        gt_data(0, T=3)
    with pytest.raises(ValueError, match="n"):
        gt_data(0, n=0)


def test_ar2_baseline_finite_and_deterministic() -> None:
    X = gt_data(0, T=100, n=8)
    e1 = ar2_baseline(X)
    e2 = ar2_baseline(X)
    assert e1 == e2
    assert np.isfinite(e1) and e1 >= 0.0


def test_ar2_baseline_rejects_short() -> None:
    with pytest.raises(ValueError, match="T"):
        ar2_baseline(np.zeros((3, 8)))


def test_ar2_beats_flat_on_structured_data() -> None:
    X = gt_data(0, T=200, n=8)
    naive = float(np.mean((X[1:] - X[:-1]) ** 2))
    assert ar2_baseline(X) < naive
