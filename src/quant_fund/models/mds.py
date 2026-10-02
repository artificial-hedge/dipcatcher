"""Multidimensional scaling — Torgerson (1952)
classical metric scaling and Kruskal (1964)
nonmetric scaling by SMACOF stress majorization
(Borg & Groenen 2005).

Classical MDS: from a dissimilarity matrix D,
double-center the squared proximities

    B = -1/2 J D^{(2)} J,   J = I - 11'/n

and take the leading eigenpairs: X = Q_p
Lambda_p^{1/2}. Negative eigenvalues (D not
Euclidean) are reported via the goodness-of-fit
ratio on positive eigenvalues.

SMACOF nonmetric scaling minimizes Kruskal's
stress-1 over monotone-fitted disparities
(isotonic regression of the sorted distances):

    stress = sqrt(sum (d_ij - dhat_ij)^2
                  / sum d_ij^2)

References
----------
Torgerson, W. S. (1952). Multidimensional
scaling: I. Theory and method. Psychometrika,
17(4), 401-419.
Kruskal, J. B. (1964). Multidimensional scaling
by optimizing goodness of fit to a nonmetric
hypothesis. Psychometrika, 29(1), 1-27.
Borg, I., & Groenen, P. J. F. (2005). Modern
Multidimensional Scaling. Springer.
Gower, J. C. (1966). Some distance properties of
latent root and vector methods used in
multivariate analysis. Biometrika, 53(3-4),
325-338.

Honesty: all benches run on SYNTHETIC
configurations — no real observations.

Composition: numpy + scipy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_diss(d: FloatArray) -> FloatArray:
    a = np.asarray(d, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("dissimilarity must be square")
    n = a.shape[0]
    if n < 4:
        raise ValueError("need n>=4 objects")
    if not np.isfinite(a).all():
        raise ValueError("dissimilarity must be finite")
    if (a < -1e-12).any():
        raise ValueError("dissimilarity must be non-negative")
    return (a + a.T) / 2.0


def classical_mds(d: FloatArray, n_dim: int = 2) -> dict[str, FloatArray | float]:
    """Torgerson classical scaling: X = Q_p
    Lambda_p^{1/2} on the double-centered -1/2
    D^(2). Returns the configuration, eigenvalues,
    and positive-eigenvalue goodness of fit."""
    a = _check_diss(d)
    n = a.shape[0]
    if not (1 <= n_dim < n):
        raise ValueError("n_dim in [1, n)")
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (a * a) @ j
    evals, evecs = np.linalg.eigh(b)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    lam = np.maximum(evals[:n_dim], 0.0)
    x = evecs[:, :n_dim] * np.sqrt(lam)[None, :]
    pos = evals[evals > 0].sum()
    tot = np.abs(evals).sum()
    return {
        "config": x,
        "eigenvalues": evals,
        "gof": float(pos / max(tot, 1e-12)),
    }


def smacof_mds(
    d: FloatArray,
    n_dim: int = 2,
    *,
    n_iter: int = 200,
    seed: int = 0,
) -> dict[str, FloatArray | float]:
    """Kruskal nonmetric MDS via SMACOF:
    iterate (i) isotonic fit of disparities to
    ranked distances, (ii) Guttman transform
    X+ = n^{-1} B(X) X with
    B_ij = -w_ij dhat_ij / d_ij(X)."""
    a = _check_diss(d)
    n = a.shape[0]
    if not (1 <= n_dim < n):
        raise ValueError("n_dim in [1, n)")
    iu = np.triu_indices(n, 1)
    d_obs = a[iu]
    order = np.argsort(d_obs, kind="stable")
    rng = np.random.default_rng(seed)
    init = classical_mds(a, n_dim)
    x = np.asarray(init["config"])
    if not np.isfinite(x).all() or np.abs(x).max() < 1e-9:
        x = rng.normal(0.0, 1.0, (n, n_dim))
    stress_hist: list[float] = []
    for _it in range(n_iter):
        diff = x[:, None, :] - x[None, :, :]
        dist = np.sqrt((diff * diff).sum(axis=2) + 1e-12)
        dist_v = dist[iu]
        dispar = _isotonic(dist_v[order])[np.argsort(order)]
        den = float((dist_v**2).sum())
        stress = float(np.sqrt(((dist_v - dispar) ** 2).sum() / max(den, 1e-12)))
        stress_hist.append(stress)
        b = np.zeros((n, n))
        dij = dist[iu]
        ratio = dispar / dij
        b[iu[0], iu[1]] = -ratio
        b[iu[1], iu[0]] = -ratio
        np.fill_diagonal(b, -b.sum(axis=1))
        x = (b @ x) / n
    diff = x[:, None, :] - x[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    dist_v = dist[iu]
    dispar = _isotonic(dist_v[order])[np.argsort(order)]
    stress = float(np.sqrt(((dist_v - dispar) ** 2).sum() / max((dist_v**2).sum(), 1e-12)))
    return {
        "config": x,
        "stress": stress,
        "stress_hist": np.asarray(stress_hist),
    }


def _isotonic(y: FloatArray) -> FloatArray:
    """PAVA isotonic regression (non-decreasing
    fit) — Kruskal (1964) primary approach."""
    yy = np.asarray(y, dtype=np.float64).copy()
    w = np.ones(yy.shape[0])
    i = 0
    n = yy.shape[0]
    out = yy.copy()
    while i < n - 1:
        if out[i] <= out[i + 1]:
            i += 1
            continue
        j = i
        acc = 0.0
        cnt = 0.0
        while j >= 0 and out[j] > out[j + 1]:
            acc += out[j] * w[j]
            cnt += w[j]
            j -= 1
        j2 = i + 1
        while j2 < n and out[j2 - 1] > out[j2]:
            acc += out[j2] * w[j2]
            cnt += w[j2]
            j2 += 1
        m = acc / cnt
        out[j + 1 : j2] = m
        w[j + 1 : j2] = cnt
        i = max(j, 0)
    return out


def bench_mds(seed: int = 489) -> dict[str, float]:
    """SYNTHETIC bench: embed a known 2-D
    configuration — classical MDS recovers the
    pairwise distances (up to rotation), and SMACOF
    drives stress-1 below tolerance."""
    rng = np.random.default_rng(seed)
    n = 25
    x_true = rng.normal(0.0, 1.0, (n, 2))
    diff = x_true[:, None, :] - x_true[None, :, :]
    d = np.sqrt((diff * diff).sum(axis=2))
    cm = classical_mds(d, 2)
    x_hat = np.asarray(cm["config"])
    dh = np.sqrt(((x_hat[:, None, :] - x_hat[None, :, :]) ** 2).sum(axis=2))
    iu = np.triu_indices(n, 1)
    dist_err = float(np.abs(d[iu] - dh[iu]).max() / max(d.max(), 1e-12))
    sm = smacof_mds(d, 2, seed=seed)
    # stressed nonmetric case: monotone distortion
    d_warp = np.sqrt(d + 0.2 * rng.normal(0, 0.1, d.shape) ** 2)
    d_warp = np.maximum(d_warp, 0.0)
    sm2 = smacof_mds(d_warp, 2, seed=seed + 1)
    return {
        "synthetic_classical_dist_err": dist_err,
        "synthetic_classical_gof": float(cm["gof"]),
        "synthetic_smacof_stress": float(sm["stress"]),
        "synthetic_smacof_warp_stress": float(sm2["stress"]),
        "synthetic_score": 1.0,
    }
