"""Manifold-learning canon: LLE (Roweis-Saul 2000), (SYNTHETIC)
Laplacian eigenmaps (Belkin-Niyogi 2003), diffusion
maps (Coifman-Lafon 2006), and exact-gradient t-SNE
(van der Maaten-Hinton 2008).

Small-n deterministic implementations for research
bench use; `bench_manifold` self-checks on a Swiss
roll (LLE/eigenmaps must unfold it so 1-NN
structure is preserved) and on two blobs for t-SNE
separation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "lle",
    "laplacian_eigenmaps",
    "diffusion_map",
    "tsne",
    "bench_manifold",
]


def _knn(x: FloatArray, k: int) -> tuple[NDArray[np.intp], FloatArray]:
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1)[:, :k]
    return idx, d


def lle(x: FloatArray, k: int = 8, d_out: int = 2) -> dict[str, object]:
    """Locally linear embedding: reconstruction
    weights then bottom eigenvectors of M=(I-W)ᵀ(I-W)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    idx, _ = _knn(x, k)
    w = np.zeros((n, n))
    tol = 1e-3
    for i in range(n):
        z = x[idx[i]] - x[i]  # (k, d)
        c = z @ z.T
        c += tol * np.trace(c) * np.eye(k)
        wv = np.linalg.solve(c, np.ones(k))
        wv /= wv.sum()
        w[i, idx[i]] = wv
    m = (np.eye(n) - w).T @ (np.eye(n) - w)
    vals, vecs = np.linalg.eigh(m)
    order = np.argsort(vals)[1 : d_out + 1]
    return {"emb": vecs[:, order], "weights": w}


def laplacian_eigenmaps(
    x: FloatArray, k: int = 8, d_out: int = 2, t: float = 0.0
) -> dict[str, object]:
    """Laplacian eigenmaps: heat-kernel W, then
    smallest generalized eigenvectors of L v = λ D v.
    t<=0 auto-scales to the median kNN squared
    distance."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    idx, d = _knn(x, k)
    if t <= 0:
        t = float(np.median(d[np.isfinite(d)]) ** 2)
    w = np.zeros((n, n))
    for i in range(n):
        w[i, idx[i]] = np.exp(-(d[i, idx[i]] ** 2) / t)
    w = np.maximum(w, w.T)
    deg = w.sum(axis=1)
    # symmetric normalized Laplacian L_sym = I − D^{-1/2} W D^{-1/2}
    dm = np.diag(1.0 / np.sqrt(deg + 1e-12))
    l_sym = np.eye(n) - dm @ w @ dm
    vals, vecs = np.linalg.eigh(l_sym)
    order = np.argsort(vals)[1 : d_out + 1]
    emb = np.diag(1.0 / np.sqrt(deg + 1e-12)) @ vecs[:, order]
    return {"emb": emb, "w": w}


def diffusion_map(
    x: FloatArray, eps: float = 0.5, d_out: int = 2, t_diff: int = 1
) -> dict[str, object]:
    """Diffusion map: row-normalized Markov kernel,
    top non-trivial right eigenvectors scaled by
    λᵗ/(1−λ)."""
    x = np.asarray(x, dtype=np.float64)
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    k_mat = np.exp(-(d**2) / (4 * eps))
    q = k_mat.sum(axis=1, keepdims=True)
    k_norm = k_mat / (q @ q.T) ** 0.5  # alpha=1 normalization
    p = k_norm / k_norm.sum(axis=1, keepdims=True)
    vals, vecs = np.linalg.eigh(p.T)
    order = np.argsort(vals)[::-1][1 : d_out + 1]
    emb = vecs[:, order] * (vals[order] ** t_diff)[None, :]
    return {"emb": emb, "eigvals": vals[order]}


def tsne(
    x: FloatArray,
    d_out: int = 2,
    perp: float = 8.0,
    it: int = 400,
    seed: int = 0,
    lr: float = 100.0,
) -> dict[str, object]:
    """t-SNE: exact pairwise affinities with binary
    perplexity search, early exaggeration 12 for the
    first 100 iters, gradient descent + momentum."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2) ** 2
    np.fill_diagonal(d, np.inf)
    logu = np.log(perp)
    p = np.zeros((n, n))
    for i in range(n):
        beta_lo, beta_hi = -np.inf, np.inf
        beta = 1.0
        di = d[i].copy()
        di[i] = 0.0
        for _ in range(60):
            pi = np.exp(-di * beta)
            h = np.log(pi.sum()) + beta * (di * pi).sum() / pi.sum()
            diff = h - logu
            if abs(diff) < 1e-5:
                break
            if diff > 0:
                beta_lo = beta
                beta = beta * 2 if np.isinf(beta_hi) else (beta + beta_hi) / 2
            else:
                beta_hi = beta
                beta = beta / 2 if np.isinf(beta_lo) else (beta + beta_lo) / 2
        pi[i] = 0.0
        p[i] = pi / pi.sum()
    p = (p + p.T) / (2 * n)
    p = np.maximum(p, 1e-12)
    rng = np.random.default_rng(seed)
    y = rng.normal(scale=1e-4, size=(n, d_out))
    y_inc = np.zeros_like(y)
    gains = np.ones_like(y)
    for it_i in range(it):
        q_num = 1.0 / (1.0 + np.linalg.norm(y[:, None, :] - y[None, :, :], axis=2) ** 2)
        np.fill_diagonal(q_num, 0.0)
        q = q_num / q_num.sum()
        q = np.maximum(q, 1e-12)
        # early exaggeration scales P only (more
        # attraction); never the repulsive term
        exagger = 12.0 if it_i < 100 else 1.0
        mult = (exagger * p - q) * q_num
        grad = 4.0 * (mult.sum(axis=1)[:, None] * y - mult @ y)
        mom = 0.5 if it_i < 100 else 0.8
        gains = (gains + 0.2) * (np.sign(grad) != np.sign(y_inc)) + (gains * 0.8) * (
            np.sign(grad) == np.sign(y_inc)
        )
        gains = np.maximum(gains, 0.01)
        y_inc = mom * y_inc - lr * gains * grad
        y = y + y_inc
        y -= y.mean(axis=0)
    return {"emb": y, "p": p}


def bench_manifold(seed: int = 536) -> dict[str, float]:
    """SYNTHETIC: Swiss roll must unfold (local
    neighbors preserved) and t-SNE must separate two
    blobs (centroid distance >> spread)."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    n = 160
    t = 3 * np.pi * rng.uniform(0.05, 0.95, n)
    swiss = np.c_[t * np.cos(t), rng.uniform(0, 5, n), t * np.sin(t)]

    def nn_preserve(emb: FloatArray) -> float:
        _, d_hi = _knn(swiss, 4)
        emb = np.asarray(emb)
        de = np.linalg.norm(emb[:, None, :] - emb[None, :, :], axis=2)
        np.fill_diagonal(de, np.inf)
        near_hi = np.argsort(d_hi, axis=1)[:, :4]
        near_lo = np.argsort(de, axis=1)[:, :4]
        ov = [len(set(near_hi[i]).intersection(near_lo[i])) / 4.0 for i in range(n)]
        return float(np.mean(ov))

    emb_l = lle(swiss, k=10, d_out=2)
    le = laplacian_eigenmaps(swiss, k=10, d_out=2)
    dm = diffusion_map(swiss, eps=5.0, d_out=2)
    for name, emb in (
        ("lle", emb_l["emb"]),
        ("le", le["emb"]),
        ("dm", dm["emb"]),
    ):
        ov = nn_preserve(np.asarray(emb))
        out[f"synthetic_swiss_{name}_nn"] = ov
        if ov < 0.4:
            raise ValueError(f"{name} unfold off: {ov:.2f}")
    # t-SNE separation on two blobs
    blobs = np.vstack([rng.normal([0, 0, 0], 0.3, (40, 3)), rng.normal([5, 5, 5], 0.3, (40, 3))])
    truth = np.repeat([0, 1], 40)
    ts = tsne(blobs, perp=8.0, it=500, seed=seed, lr=200.0)
    emb = np.asarray(ts["emb"])
    c0 = emb[truth == 0].mean(axis=0)
    c1 = emb[truth == 1].mean(axis=0)
    sep = float(np.linalg.norm(c0 - c1))
    within = float(
        np.mean(
            [
                np.linalg.norm(emb[truth == 0] - c0, axis=1).mean(),
                np.linalg.norm(emb[truth == 1] - c1, axis=1).mean(),
            ]
        )
    )
    out["synthetic_tsne_sep_ratio"] = sep / max(within, 1e-9)
    if sep / max(within, 1e-9) < 3.0:
        raise ValueError(f"tsne sep off: {sep / within:.2f}")
    return out
