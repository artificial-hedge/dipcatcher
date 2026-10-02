"""Metric learning: neighborhood components analysis (NCA,
Goldberger et al. 2004) and an LMNN-style Mahalanobis
objective (Weinberger & Saul 2009) trained by gradient
descent. Synthetic bench gates kNN gain on overlapping
blobs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def nca_learn(
    x: FloatArray,
    y: FloatArray,
    it: int = 200,
    lr: float = 0.1,
    seed: int = 0,
) -> FloatArray:
    """NCA: maximize Σ_i Σ_j p_ij [y_i = y_j] over linear map A,
    p_ij ∝ exp(−‖A x_i − A x_j‖²)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    n, d = x.shape
    rng = np.random.default_rng(seed)
    A = rng.normal(scale=0.01, size=(d, d)) + np.eye(d) * 0.5
    same = (y[:, None] == y[None, :]).astype(np.float64)
    np.fill_diagonal(same, 0.0)
    for _ in range(it):
        z = x @ A
        d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(d2, np.inf)
        p = np.exp(-d2)
        p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-16)
        np.fill_diagonal(p, 0.0)
        # f = Σ_ij p_ij·same_ij; ∂f/∂A = −2A·S/n with
        # S = Σ_ij p_ij(same_ij − f_i)x_ijx_ij' — ascent step
        w = p * (same - (p * same).sum(axis=1, keepdims=True))
        diff = x[:, None, :] - x[None, :, :]
        s = np.einsum("ij,ijd,ijk->dk", w, diff, diff)
        A -= (2 * lr / n) * (A @ s)
    return np.asarray(A)


def lmnn_learn(
    x: FloatArray,
    y: FloatArray,
    k: int = 3,
    it: int = 100,
    lr: float = 0.05,
    seed: int = 0,
) -> FloatArray:
    """LMNN-style: pull target neighbors + hinge push on
    impostors; returns Mahalanobis M = L'L via PSD gradient
    steps on the pull+push objective."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    n, d = x.shape
    np.random.default_rng(seed)
    same_mask = y[:, None] == y[None, :]
    np.fill_diagonal(same_mask, False)

    def _targets(dm2: FloatArray) -> NDArray[np.int64]:
        dm = np.where(same_mask, dm2, np.inf)
        return np.argsort(dm, axis=1)[:, :k]

    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    targets = _targets(d2)
    M = np.eye(d)
    mu = 0.5
    for t in range(it):
        # impostor hinge: for each (i, target j), push different-
        # class points inside a scale-relative margin of d(i,j)
        z = x
        dm = np.einsum("id,de,je->ij", z, M, z)
        dm2 = np.maximum(dm, 0)
        np.fill_diagonal(dm2, np.inf)
        if t % 10 == 0:
            # re-mine targets under the current metric
            targets = _targets(dm2)
        C_pull = np.zeros((d, d))
        push = np.zeros((d, d))
        for i in range(n):
            for j in targets[i]:
                diff = x[i] - x[j]
                C_pull += np.outer(diff, diff)
                d_ij = dm2[i, j]
                imp = np.flatnonzero((y != y[i]) & (dm2[i] <= d_ij * 1.25))
                for li in imp[:3]:
                    d_il = x[i] - x[li]
                    push += np.outer(d_il, d_il) - np.outer(diff, diff)
        C_pull /= max(n * k, 1)
        push /= max(n * k, 1)
        grad = C_pull - mu * push
        M = M - lr * grad
        # project to PSD
        vals, vecs = np.linalg.eigh((M + M.T) / 2)
        vals = np.maximum(vals, 1e-8)
        M = vecs @ np.diag(vals) @ vecs.T
    return np.asarray(M)


def knn_predict(
    x_tr: FloatArray, y_tr: FloatArray, x_te: FloatArray, k: int = 3, m: FloatArray | None = None
) -> FloatArray:
    """k-NN under (optional) Mahalanobis metric."""
    x_tr = np.asarray(x_tr, dtype=np.float64)
    x_te = np.asarray(x_te, dtype=np.float64)
    y_tr = np.asarray(y_tr)
    diff = x_te[:, None, :] - x_tr[None, :, :]
    if m is not None:
        d2 = np.einsum("ijd,de,ije->ij", diff, m, diff)
    else:
        d2 = (diff**2).sum(axis=2)
    nn = np.argsort(d2, axis=1)[:, :k]
    out = np.zeros(x_te.shape[0])
    for i, idx in enumerate(nn):
        vals, c = np.unique(y_tr[idx], return_counts=True)
        out[i] = vals[np.argmax(c)]
    return out


def bench_metric_learning(seed: int = 547) -> dict[str, float]:
    """SYNTHETIC, two canonical fixtures: (a) XOR blob classes —
    a diagonal-rotation Mahalanobis rescues 3NN, gated on NCA;
    (b) signal on one dim under large-scale noise dims —
    impostor-push must re-weight, gated on LMNN."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n = 240
    # fixture A: XOR class structure + noise dims
    x2 = rng.normal(0, 1.0, (n, 2))
    ya = ((x2[:, 0] > 0) ^ (x2[:, 1] > 0)).astype(np.float64)
    xa = np.c_[x2, rng.normal(0, 1.0, (n, 4))]
    perm = rng.permutation(n)
    tr, te = perm[:160], perm[160:]
    acc_raw_a = float((knn_predict(xa[tr], ya[tr], xa[te], k=3) == ya[te]).mean())
    out["synthetic_knn_xor_raw_acc"] = acc_raw_a
    A = nca_learn(xa[tr], ya[tr], it=150, lr=0.05, seed=seed)
    acc_nca = float((knn_predict(xa[tr], ya[tr], xa[te], k=3, m=A.T @ A) == ya[te]).mean())
    out["synthetic_knn_nca_acc"] = acc_nca
    # fixture B: weak signal + large-scale noise dims
    yb = rng.integers(0, 2, n).astype(np.float64)
    xb = np.c_[rng.normal(yb * 2.0, 0.6, n), rng.normal(0, 8.0, (n, 5))]
    perm = rng.permutation(n)
    tr, te = perm[:160], perm[160:]
    acc_raw_b = float((knn_predict(xb[tr], yb[tr], xb[te], k=3) == yb[te]).mean())
    out["synthetic_knn_scale_raw_acc"] = acc_raw_b
    M = lmnn_learn(xb[tr], yb[tr], k=3, it=100, lr=0.05, seed=seed)
    acc_lm = float((knn_predict(xb[tr], yb[tr], xb[te], k=3, m=M) == yb[te]).mean())
    out["synthetic_knn_lmnn_acc"] = acc_lm
    if acc_nca < acc_raw_a + 0.05:
        raise ValueError(f"nca no gain: {acc_nca} vs {acc_raw_a}")
    if acc_lm < acc_raw_b + 0.10:
        raise ValueError(f"lmnn no gain: {acc_lm} vs {acc_raw_b}")
    return out
