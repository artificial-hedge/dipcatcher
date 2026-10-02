"""Semi-supervised canon: label propagation via the harmonic function on a
k-NN similarity graph (Zhu & Ghahramani 2002) and self-training with
confidence thresholding. ``bench_semisupervised`` plants a two-moons-style
margin structure and gates both methods over the labeled-only logistic
baseline on the unlabeled pool.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import solve
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _rbf_kernel(x: FloatArray, gamma: float) -> FloatArray:
    d2 = np.sum(x * x, axis=1)[:, None] + np.sum(x * x, axis=1)[None, :] - 2.0 * (x @ x.T)
    return np.exp(-gamma * np.maximum(d2, 0.0))


def label_propagation(
    x: FloatArray,
    y: FloatArray,
    labeled: NDArray[np.bool_],
    gamma: float | None = None,
    it: int = 200,
) -> FloatArray:
    """Harmonic-function propagation; y in {0,1} for unlabeled ignored."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    if labeled.sum() < 2 or x.shape[0] != y.shape[0]:
        raise ValueError("bad inputs")
    d2 = np.sum(x * x, axis=1)[:, None] + np.sum(x * x, axis=1)[None, :] - 2.0 * (x @ x.T)
    d = np.sqrt(np.maximum(d2, 0.0))
    if gamma is None:
        gamma = 1.0 / (2.0 * np.median(d[d > 0]) ** 2)
    w = _rbf_kernel(x, gamma)
    np.fill_diagonal(w, 0.0)
    # symmetrized k-NN graph keeps the two manifolds from blurring
    k = min(7, n - 1)
    knn = np.zeros_like(w)
    nbrs = np.argsort(d, axis=1)[:, 1 : k + 1]
    rows = np.repeat(np.arange(n), k)
    knn[rows, nbrs.ravel()] = w[rows, nbrs.ravel()]
    w = np.maximum(knn, knn.T)
    deg = w.sum(axis=1)
    deg[deg <= 0] = 1.0
    # iterate f_unlab = (D_uu)^-1 (W_ul f_l + W_uu f_u)
    f = np.zeros(n)
    f[labeled] = y[labeled]
    f[~labeled] = y[labeled].mean()
    ul = ~labeled
    d_inv_u = 1.0 / deg[ul]
    for _ in range(it):
        fu = d_inv_u * (w[np.ix_(ul, labeled)] @ y[labeled] + w[np.ix_(ul, ul)] @ f[ul])
        if np.max(np.abs(fu - f[ul])) < 1e-9:
            f[ul] = fu
            break
        f[ul] = fu
    return f


def _logit_fit(x: FloatArray, y: FloatArray, lam: float, it: int = 400) -> FloatArray:
    n, d = x.shape
    w = np.zeros(d)
    for _ in range(it):
        z = np.clip(x @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = x.T @ (p - y) / n + lam * w
        h = (x.T * (p * (1 - p))) @ x / n + lam * np.eye(d)
        step = solve(h, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def _logit_p(x: FloatArray, w: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-np.clip(x @ w, -30, 30)))


def self_training(
    x: FloatArray,
    y: FloatArray,
    labeled: NDArray[np.bool_],
    threshold: float = 0.95,
    rounds: int = 10,
) -> FloatArray:
    """Iterative self-training: add confident predictions as pseudo-labels."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lab = labeled.copy()
    yw = y.copy()
    for _ in range(rounds):
        w = _logit_fit(x[lab], yw[lab], lam=1e-3)
        p = _logit_p(x, w)
        conf = np.maximum(p, 1.0 - p)
        new = (~lab) & (conf >= threshold)
        if not new.any():
            break
        yw[new] = (p[new] >= 0.5).astype(np.float64)
        lab |= new
    return _logit_p(x, _logit_fit(x[lab], yw[lab], lam=1e-3))


def bench_semisupervised(seed: int = 20261231) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    # Two moons: nonlinear boundary, unlabeled manifold structure.
    n = 240
    cls = (rng.uniform(0, 1, n) > 0.5).astype(np.float64)
    t = rng.uniform(0, np.pi, n)
    x = np.zeros((n, 2))
    m0 = cls == 0
    x[m0, 0] = np.cos(t[m0])
    x[m0, 1] = np.sin(t[m0])
    x[~m0, 0] = 1.0 - np.cos(t[~m0])
    x[~m0, 1] = -np.sin(t[~m0]) + 0.5
    x += 0.08 * rng.standard_normal(x.shape)
    y = cls
    perm = rng.permutation(n)
    lab_mask = np.zeros(n, dtype=bool)
    lab_mask[perm[:16]] = True  # ~7% labeled
    f_lp = label_propagation(x, y, lab_mask)
    p_st = self_training(x, y, lab_mask, threshold=0.9)
    w0 = _logit_fit(x[lab_mask], y[lab_mask], lam=1e-3)
    p0 = _logit_p(x, w0)
    unl = ~lab_mask
    acc = lambda p: float(  # noqa: E731
        np.mean((p[unl] >= 0.5) == (y[unl] >= 0.5))
    )
    return {
        "synthetic_base_acc": acc(p0),
        "synthetic_labelprop_acc": acc(f_lp),
        "synthetic_selftrain_acc": acc(p_st),
        "synthetic_lp_gain": acc(f_lp) - acc(p0),
    }
