"""Tensor power-iteration canon: robust symmetric tensor decomposition
(Anandkumar et al. 2014) — whitening the empirical third moment of a
spherical-Gaussian mixture / topic-style latent model and recovering the
factor vectors via deflated tensor power with random restarts.
``bench_tensor_power`` plants k orthogonal factor vectors inside a
low-rank-plus-noise third moment and gates recovery cosine similarity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _tvt(t: FloatArray, v: FloatArray) -> float:
    return float(np.einsum("ijk,i,j,k->", t, v, v, v))


def _tvc(t: FloatArray, v: FloatArray) -> FloatArray:
    return np.asarray(np.einsum("ijk,j,k->i", t, v, v))


def tensor_power_iteration(
    t: FloatArray, n_init: int = 20, it: int = 100, seed: int = 0
) -> tuple[FloatArray, float]:
    """Best (v, lambda) pair: v <- T(I,v,v)/||.|| from random restarts."""
    t = np.asarray(t, dtype=np.float64)
    d = t.shape[0]
    if t.shape != (d, d, d):
        raise ValueError("t must be cubic")
    rng = np.random.default_rng(seed)
    best_v = np.zeros(d)
    best_l = -np.inf
    for _ in range(n_init):
        v = np.asarray(rng.standard_normal(d), dtype=np.float64)
        v = np.asarray(v / float(np.linalg.norm(v)), dtype=np.float64)
        for _ in range(it):
            tv = _tvc(t, v)
            nv = np.linalg.norm(tv)
            if nv <= 1e-12:
                break
            v_new = tv / nv
            if np.max(np.abs(v_new - v)) < 1e-10:
                v = v_new
                break
            v = v_new
        lam = _tvt(t, v)
        if lam > best_l:
            best_l, best_v = lam, v
    return best_v, float(best_l)


def tensor_deflation(
    t: FloatArray,
    k: int,
    n_init: int = 20,
    it: int = 100,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Sequential power + deflation: returns (vectors (k,d), lambdas (k))."""
    t = np.asarray(t, dtype=np.float64).copy()
    d = t.shape[0]
    vecs = np.zeros((k, d))
    lams = np.zeros(k)
    for i in range(k):
        v, lam = tensor_power_iteration(t, n_init=n_init, it=it, seed=seed + i)
        vecs[i] = v
        lams[i] = lam
        t = t - lam * np.einsum("i,j,k->ijk", v, v, v)
    return vecs, lams


def whiten(
    x: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Second-moment whitening: returns (W (k,d), x_white (n,k)) with
    x_white @ x_white' / n = I_k."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    mu = x.mean(axis=0)
    xc = x - mu
    cov = xc.T @ xc / n
    vals, vecs = np.linalg.eigh(cov)
    keep = vals > 1e-10 * vals.max()
    vals_k = vals[keep]
    vecs_k = vecs[:, keep]
    w = np.diag(1.0 / np.sqrt(vals_k)) @ vecs_k.T
    return w, xc @ w.T


def third_moment_whitened(xw: FloatArray) -> FloatArray:
    n = xw.shape[0]
    return np.asarray(np.einsum("ni,nj,nk->ijk", xw, xw, xw) / n)


def bench_tensor_power(seed: int = 20261231) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n, d, k = 4000, 6, 3
    # spherical GMM: component means mu_c (orthogonal-ish), x = mu_c + N(0,I)
    base = rng.standard_normal((k, d))
    q, _ = np.linalg.qr(base.T)
    mus = q[:, :k].T * 2.5
    z = rng.integers(0, k, n)
    x = mus[z] + rng.standard_normal((n, d))
    w, xw = whiten(x)
    t3 = third_moment_whitened(xw)
    vecs, lams = tensor_deflation(t3, k, n_init=30, seed=seed)
    # map whitened factors back and compare to true whitened mus
    xbar = x.mean(axis=0)
    true_w = (mus - xbar) @ w.T  # centered means in whitened coords
    true_w = true_w / np.linalg.norm(true_w, axis=1, keepdims=True)
    sim = np.abs(vecs @ true_w.T)
    best = sim.max(axis=1)
    return {
        "synthetic_tp_cos_min": float(best.min()),
        "synthetic_tp_cos_mean": float(best.mean()),
        "synthetic_tp_lam_sum": float(np.abs(lams).sum()),
        "synthetic_tp_resid": float(
            np.linalg.norm(
                t3
                - sum(
                    lams[i] * np.einsum("a,b,c->abc", vecs[i], vecs[i], vecs[i]) for i in range(k)
                )
            )
            / np.linalg.norm(t3)
        ),
    }
