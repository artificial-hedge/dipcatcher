"""Ripley's K/L spatial point-pattern analysis.

Ripley (1976, 1977): for a homogeneous planar Poisson process of
intensity lambda, K(r) = (1/lambda) E[# extra points within distance r
of a typical point] = pi r^2. The variance-stabilized transform
L(r) = sqrt(K(r)/pi) satisfies L(r) - r = 0 under CSR; positive
deviations indicate clustering, negative regularity. Estimation uses
isotropic (Ripley) edge correction: pairs crossing the window boundary
are weighted by the fraction of the r-circle inside the window.

The Monte Carlo envelope (Diggle 2003) simulates B CSR patterns with
the same n and window; a point lying above the upper envelope is
significant clustering at approximate level 2/(B+1) per distance — an
honest simultaneous-band caveat, not a per-r guarantee.

Honesty: 2D only, rectangular window, border-corrected. The bench
verifies L-hat - r ~ 0 on CSR and > 0 on a Matern cluster process.
Fail-closed on empty patterns, non-finite coords, or r beyond the
window quarter-diagonal (edge corrections blow up).

References: Ripley (1976) "The second-order analysis of stationary
point processes"; Baddeley, Rubak & Turner (2015) Spatial Point
Patterns: Methodology and Applications with R; Dixon (2014) in
Encyclopedia of Statistical Sciences.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _distmat(pts: FloatArray) -> FloatArray:
    d = pts[:, None, :] - pts[None, :, :]
    return np.asarray(np.sqrt((d * d).sum(axis=2)), dtype=np.float64)


def ripley_k(
    points: FloatArray,
    window: tuple[float, float, float, float],
    r_grid: FloatArray,
) -> FloatArray:
    """Ripley's K-hat on [x0,x1]x[y0,y1] with isotropic edge correction.

    K_hat(r) = (A / n^2) sum_{i != j} 1[d_ij <= r] / w_ij,

    where A is window area and w_ij is the in-window fraction of the
    circle of radius d_ij centred on point i.
    """
    pts = np.asarray(points, dtype=float)
    x0, x1, y0, y1 = window
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 10:
        raise ValueError("need (n>=10, 2) points")
    if not (x0 < x1 and y0 < y1):
        raise ValueError("bad window")
    r = np.asarray(r_grid, dtype=float).ravel()
    if r.size < 2 or np.any(np.diff(r) <= 0) or r[0] <= 0:
        raise ValueError("bad r grid")
    diag = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    if r[-1] > 0.25 * diag:
        raise ValueError("r grid exceeds quarter-diagonal edge limit")
    inside = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
    if not inside.all():
        raise ValueError("points outside window")
    n = pts.shape[0]
    d = _distmat(pts)
    np.fill_diagonal(d, np.inf)
    # border distance for each point
    bord = np.minimum.reduce([pts[:, 0] - x0, x1 - pts[:, 0], pts[:, 1] - y0, y1 - pts[:, 1]])
    # w_ij: fraction of the circle of radius d_ij around point i that
    # lies inside the window. For d_ij <= bord_i the whole circle is
    # inside (w=1); beyond it the standard isotropic segment correction
    # drops the clipped fraction 1 - (2/pi) arccos(bord/d).
    w_ij = np.ones_like(d)
    for i in range(n):
        di = d[i]
        bi = bord[i]
        big = di > bi
        if not np.any(big):
            continue
        frac = np.ones(n)
        frac[big] = 1.0 - (2.0 / np.pi) * np.arccos(np.clip(bi / di[big], -1.0, 1.0))
        w_ij[i, big] = frac[big]
    w_ij = np.maximum(w_ij, 0.25)
    a = (x1 - x0) * (y1 - y0)
    k = np.empty(r.size)
    for j, rr in enumerate(r):
        cnt = (d <= rr) / w_ij
        k[j] = a * cnt.sum() / (n * n)
    return np.asarray(k, dtype=np.float64)


def ripley_l(k: FloatArray) -> FloatArray:
    """L(r) = sqrt(K(r)/pi), the variance-stabilized form."""
    k = np.asarray(k, dtype=float).ravel()
    if np.any(k < 0):
        raise ValueError("negative K")
    return np.asarray(np.sqrt(k / np.pi), dtype=np.float64)


def csr_envelope(
    n: int,
    window: tuple[float, float, float, float],
    r_grid: FloatArray,
    n_sim: int = 39,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """(lo, hi) L-hat envelopes from n_sim CSR realizations."""
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = window
    r = np.asarray(r_grid, dtype=float).ravel()
    sims = np.empty((n_sim, r.size))
    for b in range(n_sim):
        pts = np.column_stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)])
        sims[b] = ripley_l(ripley_k(pts, window, r))
    return (
        np.asarray(sims.min(axis=0), dtype=np.float64),
        np.asarray(sims.max(axis=0), dtype=np.float64),
    )


def _matern_cluster(
    n_parents: int, spread: float, n_children: int, rng: np.random.Generator
) -> FloatArray:
    par = rng.uniform(0.0, 1.0, (n_parents, 2))
    pts = []
    for p in par:
        ang = rng.uniform(0, 2 * np.pi, n_children)
        rad = spread * np.sqrt(rng.uniform(0, 1, n_children))
        pts.append(p + np.column_stack([rad * np.cos(ang), rad * np.sin(ang)]))
    pts_arr = np.concatenate(pts)
    inside = (pts_arr >= 0.0).all(axis=1) & (pts_arr <= 1.0).all(axis=1)
    return np.asarray(pts_arr[inside], dtype=np.float64)


def bench_ripley_k(seed: int = 20261231 + 400) -> dict[str, float]:
    """SYNTHETIC check — CSR sits inside envelope; Matern clusters exceed it."""
    rng = np.random.default_rng(seed)
    window = (0.0, 1.0, 0.0, 1.0)
    r = np.linspace(0.01, 0.2, 12)
    csr = rng.uniform(0.0, 1.0, (200, 2))
    l_csr = ripley_l(ripley_k(csr, window, r))
    dev_csr = float(np.max(np.abs(l_csr - r)))
    if dev_csr > 0.04:
        raise ValueError(f"CSR deviation too large: {dev_csr}")
    clu = _matern_cluster(20, 0.04, 12, rng)
    l_clu = ripley_l(ripley_k(clu, window, r))
    dev_clu = float(np.max(l_clu - r))
    if dev_clu < 0.03:
        raise ValueError(f"cluster deviation too small: {dev_clu}")
    lo, hi = csr_envelope(200, window, r, n_sim=39, seed=seed + 2)
    inside_frac = float(np.mean((l_csr >= lo) & (l_csr <= hi)))
    above_frac = float(np.mean(l_clu > hi))
    if inside_frac < 0.8:
        raise ValueError("CSR escapes envelope too often")
    if above_frac < 0.5:
        raise ValueError("cluster not detected above envelope")
    return {
        "synthetic_rk_csr_dev": dev_csr,
        "synthetic_rk_cluster_dev": dev_clu,
        "synthetic_rk_csr_in_env": inside_frac,
        "synthetic_rk_cluster_above_env": above_frac,
        "synthetic_score": 1.0,
    }
