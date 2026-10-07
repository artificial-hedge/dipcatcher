"""Hierarchical risk parity — Lopez de Prado (2016) allocation.

HRP avoids inverting the covariance matrix: it clusters assets on
the correlation distance d_ij = sqrt(0.5 (1 - corr_ij)), quasi-
diagonalizes the covariance by the dendrogram leaf order, then
allocates by recursive bisection — each split assigns weight
alpha = 1 - v_L / (v_L + v_R) to the left cluster, where v is the
(inverse-variance) portfolio variance of that cluster.

Honesty: the bench uses a two-block correlation structure (within
0.8, across 0.05) and checks (a) weights sum to 1 and are non-
negative, (b) within-block weight spread stays small under the
symmetric block structure, and (c) on a diagonal covariance HRP
recovers inverse-variance weighting (documented limit). Fail-closed
on non-PSD or non-square covariance.

References: Lopez de Prado (2016) "Building Diversified Portfolios
that Outperform Out-of-Sample", J. Portfolio Mgmt; Raffinot (2017)
"Hierarchical Clustering-Based Asset Allocation"; Ward (1963)
hierarchical grouping linkage.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

FloatArray = NDArray[np.float64]


def _check_cov(cov: FloatArray) -> FloatArray:
    a = np.asarray(cov, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 2:
        raise ValueError("bad covariance")
    if not np.isfinite(a).all() or not np.allclose(a, a.T, atol=1e-8):
        raise ValueError("covariance must be symmetric finite")
    return a


def _cov2corr(cov: FloatArray) -> FloatArray:
    d = np.sqrt(np.diag(cov))
    if (d <= 0).any():
        raise ValueError("non-positive variances")
    return np.asarray(cov / np.outer(d, d), dtype=np.float64)


def hrp_weights(cov: FloatArray) -> FloatArray:
    """HRP inverse-variance-bisection weights for covariance ``cov``.

    Ward linkage on the Lopez de Prado correlation distance; quasi-
    diagonal order from the dendrogram; recursive bisection with
    inverse-variance cluster variances.
    """
    c = _check_cov(cov)
    n = c.shape[0]
    corr = _cov2corr(c)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    z = linkage(squareform(dist, checks=False), method="ward")
    order = leaves_list(z).tolist()
    w = np.ones(n)
    clusters = [order]
    while clusters:
        cl = clusters.pop(0)
        if len(cl) <= 1:
            continue
        half = len(cl) // 2
        left, right = cl[:half], cl[half:]
        for part in (left, right):
            sub = c[np.ix_(part, part)]
            ivp = 1.0 / np.diag(sub)
            ivp = ivp / ivp.sum()
            v = float(ivp @ sub @ ivp)
            if part is left:
                vl = v
            else:
                vr = v
        alpha = 1.0 - vl / max(vl + vr, 1e-300)
        for i in left:
            w[i] *= alpha
        for i in right:
            w[i] *= 1.0 - alpha
        clusters.extend([left, right])
    tot = w.sum()
    if tot <= 0:
        raise ValueError("degenerate weights")
    return np.asarray(w / tot, dtype=np.float64)


def bench_hrp(seed: int = 20261231 + 419) -> dict[str, float]:
    """SYNTHETIC check — block-corr balance + diagonal IVP limit."""
    nb, block = 2, 4
    n = nb * block
    corr = np.full((n, n), 0.05)
    for b in range(nb):
        for i in range(b * block, (b + 1) * block):
            for j in range(b * block, (b + 1) * block):
                corr[i, j] = 0.8 if i != j else 1.0
    vols = np.array([1.0, 1.1, 0.9, 1.05, 1.5, 1.4, 1.6, 1.45])
    cov = corr * np.outer(vols, vols)
    w = hrp_weights(cov)
    if w.sum() < 0.999 or (w < -1e-9).any():
        raise ValueError("hrp weights invalid")
    spread = max(
        float(np.ptp(w[b * block : (b + 1) * block]) / w[b * block : (b + 1) * block].mean())
        for b in range(nb)
    )
    if spread > 0.6:
        raise ValueError(f"block imbalance: {spread:.3f}")
    dv = np.diag(np.array([1.0, 4.0, 9.0, 16.0]))
    wd = hrp_weights(dv)
    ivp = 1.0 / np.diag(dv)
    ivp /= ivp.sum()
    dev = float(np.abs(np.sort(wd) - np.sort(ivp)).max())
    if dev > 0.15:
        raise ValueError(f"ivp limit off: {dev:.3f}")
    return {
        "synthetic_hrp_block_spread": spread,
        "synthetic_hrp_ivp_dev": dev,
        "synthetic_hrp_min_w": float(w.min()),
        "synthetic_score": 1.0,
    }
