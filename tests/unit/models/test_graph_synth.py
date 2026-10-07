"""Unit tests for quant_fund.models._graph_synth."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models._graph_synth import normalize_adj, split_masks, synth_sbm_graph


def test_sbm_graph_shapes_symmetric_deterministic() -> None:
    a1, x1, y1 = synth_sbm_graph(n=60, seed=0)
    a2, x2, y2 = synth_sbm_graph(n=60, seed=0)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(y1, y2)
    assert a1.shape == (60, 60)
    np.testing.assert_array_equal(a1, a1.T)
    assert (np.diag(a1) == 0).all()
    assert x1.shape == (60, 16)
    assert set(np.unique(y1)) <= set(range(4))


def test_sbm_in_density_exceeds_out() -> None:
    adj, _x, y = synth_sbm_graph(n=200, seed=1)
    same = y[:, None] == y[None, :]
    off = ~np.eye(200, dtype=bool)
    p_in_hat = adj[same & off].mean()
    p_out_hat = adj[~same & off].mean()
    assert p_in_hat > 3 * p_out_hat


def test_sbm_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="p_in"):
        synth_sbm_graph(p_in=1.5)
    with pytest.raises(ValueError, match="p_out"):
        synth_sbm_graph(p_out=-0.1)
    with pytest.raises(ValueError, match="feat_noise"):
        synth_sbm_graph(feat_noise=-1.0)
    with pytest.raises(ValueError, match="n"):
        synth_sbm_graph(n=0)


def test_normalize_adj_row_scale() -> None:
    adj, _x, _y = synth_sbm_graph(n=40, seed=0)
    an = normalize_adj(adj)
    assert an.shape == adj.shape
    np.testing.assert_allclose(an, an.T, atol=1e-12)
    assert np.isfinite(an).all()
    # isolated-safe: disconnected node survives through self-loop
    an_ns = normalize_adj(adj, self_loops=False)
    assert np.isfinite(an_ns).all()


def test_normalize_adj_rejects_negative_edges() -> None:
    bad = np.array([[0.0, -1.0], [-1.0, 0.0]])
    with pytest.raises(ValueError, match="non-negative"):
        normalize_adj(bad)


def test_split_masks_partition() -> None:
    tr, te = split_masks(100, 0.4, seed=0)
    assert tr.sum() == 40 and te.sum() == 60
    assert not (tr & te).any()
    assert (tr | te).all()
    tr2, _te2 = split_masks(100, 0.4, seed=0)
    np.testing.assert_array_equal(tr, tr2)


def test_split_masks_rejects_out_of_range_frac() -> None:
    # frac>1 silently gives an empty test mask (vacuous eval); frac<0
    # mislabels nearly all nodes as train. Both must raise.
    with pytest.raises(ValueError, match="frac_train"):
        split_masks(100, 1.5, seed=0)
    with pytest.raises(ValueError, match="frac_train"):
        split_masks(100, -0.2, seed=0)
    with pytest.raises(ValueError, match="n"):
        split_masks(0, 0.5, seed=0)
