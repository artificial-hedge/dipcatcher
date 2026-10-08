"""Normalized spectral clustering (Ng, Jordan & Weiss 2002) (SYNTHETIC).

The algorithm builds a Gaussian affinity graph, forms the symmetric normalized
Laplacian ``L_sym = I - D^{-1/2} W D^{-1/2}``, embeds points using its ``k``
smallest eigenvectors, row-normalises the embedding, and clusters the rows with
k-means.  It recovers non-convex clusters that k-means on the raw data cannot.

Reference: A. Ng, M. Jordan, Y. Weiss (2002), "On spectral clustering: analysis
and an algorithm", NeurIPS.  Fail-closed on invalid shapes or parameters.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

Array = NDArray[np.float64]
IntArray = NDArray[np.int_]


def _median_sigma(x: Array) -> float:
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))
    iu = np.triu_indices(x.shape[0], k=1)
    med = float(np.median(dist[iu]))
    return med if med > 0 else 1.0


def spectral_clustering(
    x: Array, n_clusters: int, sigma: float | None = None, seed: int = 0
) -> dict[str, IntArray | Array]:
    """Cluster rows of ``x`` with normalized spectral clustering."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < n_clusters or not np.isfinite(arr).all():
        raise ValueError("x must be a finite (n, d) matrix with n >= n_clusters")
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2")
    s = _median_sigma(arr) if sigma is None else float(sigma)
    if s <= 0.0:
        raise ValueError("sigma must be positive")
    diff = arr[:, None, :] - arr[None, :, :]
    sq = (diff**2).sum(axis=2)
    w = np.exp(-sq / (2.0 * s**2))
    np.fill_diagonal(w, 0.0)
    d = w.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.where(d > 0, d, 1.0))
    l_sym = np.eye(arr.shape[0]) - (d_inv_sqrt[:, None] * w * d_inv_sqrt[None, :])
    eigvals, eigvecs = np.linalg.eigh(l_sym)
    embed = eigvecs[:, :n_clusters]
    norms = np.linalg.norm(embed, axis=1, keepdims=True)
    embed = embed / np.where(norms > 0, norms, 1.0)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(embed)
    return {"labels": labels.astype(int), "eigenvalues": eigvals[:n_clusters]}


def bench_spectral_clustering(seed: int = 0) -> dict[str, float]:
    """Spectral-clustering oracle: well-separated blobs must be recovered
    almost perfectly; the labels must be deterministic."""
    rng = np.random.default_rng(seed)
    c1 = rng.standard_normal((40, 2)) * 0.4 + np.array([0.0, 0.0])
    c2 = rng.standard_normal((40, 2)) * 0.4 + np.array([4.0, 4.0])
    c3 = rng.standard_normal((40, 2)) * 0.4 + np.array([0.0, 4.5])
    x = np.vstack([c1, c2, c3])
    truth = np.repeat([0, 1, 2], 40)
    out = spectral_clustering(x, 3, seed=seed)
    lab = np.asarray(out["labels"])
    # purity: majority-vote agreement up to label permutation
    best = 0.0
    from itertools import permutations

    for perm in permutations(range(3)):
        best = max(
            best, float((lab == np.vectorize({i: p for i, p in enumerate(perm)}.get)(truth)).mean())
        )
    out_b = spectral_clustering(x, 3, seed=seed)
    same = float((np.asarray(out_b["labels"]) == lab).mean())
    checks = [best > 0.95, same == 1.0]
    if not all(checks):
        raise ValueError("spectral-clustering purity/determinism oracle failed")
    return {
        "synthetic_sclust_purity": best,
        "synthetic_sclust_determinism": same,
        "synthetic_spectral_clustering_score": float(sum(bool(c) for c in checks) / len(checks)),
    }
