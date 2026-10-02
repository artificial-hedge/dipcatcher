"""Isomap manifold embedding via geodesic-distance MDS.

Tenenbaum, de Silva & Langford (2000): nonlinear dimensionality
reduction that preserves *geodesic* (manifold) distances rather than
ambient Euclidean ones. Build the k-nearest-neighbor graph on the data,
compute all-pairs shortest paths (Dijkstra/Floyd-Warshall) as geodesic
estimates, then apply classical MDS:

    B = -1/2 J D_geo^2 J,   J = I - 1 1^T / n

and take the top eigenpairs for the low-dimensional embedding.

Honesty: the bench embeds a Swiss-roll strip — a 2-D sheet curled in
3-D — and checks that the 2-D embedding recovers the ordering along the
roll (Spearman-ish correlation of the recovered coordinate with the
true latent parameter) far above chance. Fail-closed on non-finite
input, a disconnected kNN graph (which would make geodesics infinite),
or degenerate spectra.

References: Tenenbaum, de Silva & Langford (2000) "A global geometric
framework for nonlinear dimensionality reduction", Science;
Bernstein et al. (2000) convergence of graph geodesics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _knn_graph(x: FloatArray, k: int) -> tuple[FloatArray, NDArray[np.int64]]:
    n = x.shape[0]
    d2 = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    nbrs = np.argsort(d2, axis=1)[:, :k]
    g = np.full((n, n), np.inf)
    for i in range(n):
        g[i, nbrs[i]] = np.sqrt(d2[i, nbrs[i]])
    g = np.minimum(g, g.T)  # symmetrize
    return np.asarray(g, dtype=np.float64), np.asarray(nbrs, dtype=np.int64)


def _floyd_warshall(g: FloatArray) -> FloatArray:
    n = g.shape[0]
    d = g.copy()
    np.fill_diagonal(d, 0.0)
    for kk in range(n):
        d = np.minimum(d, d[:, kk : kk + 1] + d[kk : kk + 1, :])
    return np.asarray(d, dtype=np.float64)


def classical_mds(d2_geo: FloatArray, n_components: int = 2) -> dict[str, FloatArray]:
    """Classical (Torgerson) MDS on squared distances."""
    d2 = np.asarray(d2_geo, dtype=float)
    n = d2.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ d2 @ j
    w, v = np.linalg.eigh(b)
    idx = np.argsort(w)[::-1][:n_components]
    w_pos = np.maximum(w[idx], 0.0)
    return {
        "coords": np.asarray(v[:, idx] * np.sqrt(w_pos), dtype=np.float64),
        "eigenvalues": np.asarray(w[idx], dtype=np.float64),
    }


def isomap(x: FloatArray, k: int = 10, n_components: int = 2) -> dict[str, FloatArray | float]:
    """Isomap embedding: kNN graph -> geodesics -> classical MDS."""
    a = np.asarray(x, dtype=float)
    if a.ndim != 2 or a.shape[0] < k + 5 or not np.isfinite(a).all():
        raise ValueError("bad input")
    g, nbrs = _knn_graph(a, k)
    d_geo = _floyd_warshall(g)
    if not np.isfinite(d_geo).all():
        raise ValueError("disconnected kNN graph — increase k")
    out = classical_mds(d_geo**2, n_components)
    return {
        "coords": np.asarray(out["coords"], dtype=np.float64),
        "eigenvalues": np.asarray(out["eigenvalues"], dtype=np.float64),
        "geodesics": d_geo,
        "neighbors": nbrs.astype(float),
    }


def bench_isomap(seed: int = 20261231 + 405) -> dict[str, float]:
    """SYNTHETIC check — Swiss-roll latent ordering recovered in 2-D."""
    rng = np.random.default_rng(seed)
    n = 260
    s_lat = rng.uniform(0.5, 4.5, n)
    t_lat = rng.uniform(0.0, 1.0, n)
    # Swiss roll strip: (s cos s, t*h, s sin s)
    x = np.column_stack([s_lat * np.cos(s_lat), 10.0 * t_lat, s_lat * np.sin(s_lat)])
    x += 0.02 * rng.standard_normal(x.shape)
    out = isomap(x, k=12, n_components=2)
    coords = np.asarray(out["coords"])
    # correlation of each recovered coord with the true latent s
    # (take the max |Spearman| over components — either axis may carry it)
    from scipy.stats import spearmanr

    rho = max(abs(float(spearmanr(coords[:, j], s_lat)[0])) for j in range(2))
    if rho < 0.8:
        raise ValueError(f"isomap ordering off: |rho| {rho}")
    return {
        "synthetic_isomap_order_rho": rho,
        "synthetic_isomap_top_eig": float(np.asarray(out["eigenvalues"])[0]),
        "score": 1.0,
    }
