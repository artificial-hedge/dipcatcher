"""Manifold learning: LLE, eigenmaps, diffusion, t-SNE."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.manifold_learning import (
    diffusion_map,
    laplacian_eigenmaps,
    lle,
    tsne,
)


@pytest.fixture()
def swiss() -> np.ndarray:
    rng = np.random.default_rng(11)
    n = 120
    t = 3 * np.pi * rng.uniform(0.05, 0.95, n)
    return np.c_[t * np.cos(t), rng.uniform(0, 5, n), t * np.sin(t)]


def test_lle_reconstruction_weights(swiss):
    r = lle(swiss, k=10, d_out=2)
    w = np.asarray(r["weights"])
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)
    assert np.asarray(r["emb"]).shape == (len(swiss), 2)


def test_lle_unrolls(swiss):
    r = lle(swiss, k=10, d_out=2)
    emb = np.asarray(r["emb"])
    assert np.linalg.det(np.cov(emb.T)) > 1e-6


def test_laplacian_eigenmaps_shape(swiss):
    r = laplacian_eigenmaps(swiss, k=10, d_out=2)
    assert np.asarray(r["emb"]).shape == (len(swiss), 2)


def test_diffusion_map_eigvals(swiss):
    r = diffusion_map(swiss, eps=5.0, d_out=3)
    ev = np.asarray(r["eigvals"])
    assert ev[0] <= 1.0 + 1e-9
    assert np.asarray(r["emb"]).shape == (len(swiss), 3)


def test_tsne_separates_blobs():
    rng = np.random.default_rng(5)
    x = np.vstack([rng.normal([0, 0, 0], 0.3, (30, 3)), rng.normal([6, 6, 6], 0.3, (30, 3))])
    emb = np.asarray(tsne(x, perp=8.0, it=500, seed=1, lr=200.0)["emb"])
    c0, c1 = emb[:30].mean(0), emb[30:].mean(0)
    assert np.linalg.norm(c0 - c1) > 0.5
