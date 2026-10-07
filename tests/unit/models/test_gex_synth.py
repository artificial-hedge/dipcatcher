"""Unit tests for quant_fund.models._gex_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._gex_synth import planted_clique


def test_adjacency_is_symmetric_undirected() -> None:
    A, x, y = planted_clique(seed=7)
    np.testing.assert_array_equal(A, A.T)  # undirected graph
    assert (np.diag(A) == 0).all()
    assert set(np.unique(A)) <= {0.0, 1.0}


def test_clique_nodes_fully_connected() -> None:
    A, _x, y = planted_clique(seed=7, n=16, k=5)
    members = np.flatnonzero(y)
    assert members.size == 5
    sub = A[np.ix_(members, members)]
    assert (sub + np.eye(5) == 1.0).all()  # every pair connected


def test_labels_match_planted_set() -> None:
    _A, _x, y = planted_clique(seed=11)
    assert y.sum() == 5.0
    assert set(np.unique(y)) <= {0.0, 1.0}


def test_determinism() -> None:
    A1, x1, y1 = planted_clique(seed=42)
    A2, x2, y2 = planted_clique(seed=42)
    np.testing.assert_array_equal(A1, A2)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)


def test_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="p"):
        planted_clique(p=1.5)
    with pytest.raises(ValueError, match="p"):
        planted_clique(p=-0.1)
    with pytest.raises(ValueError, match="k"):
        planted_clique(k=1)


def test_features_shape() -> None:
    _A, x, _y = planted_clique(seed=0, n=16)
    assert x.shape == (16, 4)
    assert np.isfinite(x).all()
