"""Sparse coding: orthogonal matching pursuit (Pati 1993)
and K-SVD dictionary learning (Aharon-Elad-Bruckstein
2006). Synthetic bench gates OMP support recovery and
K-SVD reconstruction of signals generated from a planted
dictionary."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def omp(dictionary: FloatArray, y: FloatArray, n_atoms: int, tol: float = 1e-6) -> IntArray:
    """OMP: greedily pick max-correlation atom, LS-solve on
    the active set. Returns selected atom indices."""
    d = np.asarray(dictionary, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    active: list[int] = []
    resid = y.copy()
    for _ in range(min(n_atoms, d.shape[1])):
        corr = np.abs(d.T @ resid)
        corr[active] = -np.inf
        active.append(int(np.argmax(corr)))
        coef, *_ = np.linalg.lstsq(d[:, active], y, rcond=None)
        resid = y - d[:, active] @ coef
        if float(resid @ resid) < tol**2:
            break
    return np.asarray(sorted(active), dtype=np.int64)


def omp_solve(dictionary: FloatArray, y: FloatArray, n_atoms: int) -> FloatArray:
    """OMP support + LS coefficients; returns sparse coef."""
    d = np.asarray(dictionary, dtype=np.float64)
    idx = omp(d, y, n_atoms)
    coef = np.zeros(d.shape[1])
    if len(idx):
        coef[idx], *_ = np.linalg.lstsq(d[:, idx], y, rcond=None)
    return np.asarray(coef)


def ksvd(
    x: FloatArray,
    n_atoms: int,
    sparsity: int,
    it: int = 20,
    seed: int = 0,
) -> dict[str, object]:
    """K-SVD: alternate OMP coding with atom-by-atom SVD
    updates on the residual restricted to each atom's users."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    n, p = x.shape[1], x.shape[0]
    d = rng.normal(0, 1, (p, n_atoms))
    d /= np.linalg.norm(d, axis=0, keepdims=True)
    gamma = np.zeros((n_atoms, n))
    for _ in range(it):
        for i in range(n):
            gamma[:, i] = omp_solve(d, x[:, i], sparsity)
        for k in range(n_atoms):
            users = np.where(np.abs(gamma[k]) > 1e-12)[0]
            if len(users) == 0:
                d[:, k] = rng.normal(0, 1, p)
                d[:, k] /= np.linalg.norm(d[:, k])
                continue
            err = x[:, users] - d @ gamma[:, users] + np.outer(d[:, k], gamma[k, users])
            u, sv, vt = np.linalg.svd(err, full_matrices=False)
            d[:, k] = u[:, 0]
            gamma[k, users] = sv[0] * vt[0]
    return {"dictionary": d, "codes": gamma}


def bench_sparse_coding(seed: int = 562) -> dict[str, float]:
    """SYNTHETIC: signals = D_true @ sparse codes; OMP must
    recover the planted support, K-SVD must reconstruct to
    low residual."""
    rng = np.random.default_rng(seed)
    out: dict[str, float] = {}
    p, k, n = 24, 16, 120
    d_true = rng.normal(0, 1, (p, k))
    d_true /= np.linalg.norm(d_true, axis=0, keepdims=True)
    codes = np.zeros((k, n))
    supports: list[IntArray] = []
    for i in range(n):
        sup = rng.choice(k, 3, replace=False)
        supports.append(np.asarray(sorted(sup)))
        codes[sup, i] = rng.normal(0, 1, 3)
    x = d_true @ codes + rng.normal(0, 0.02, (p, n))
    # OMP support recovery
    hits = 0
    for i in range(30):
        est = omp(d_true, x[:, i], 3)
        hits += int(len(set(est.tolist()) & set(supports[i].tolist())))
    out["synthetic_omp_support_hit"] = float(hits)
    if hits < 80:  # of 90
        raise ValueError(f"omp support off: {hits}")
    # K-SVD reconstruction
    mdl = ksvd(x, n_atoms=k, sparsity=3, it=25, seed=seed)
    d_hat = np.asarray(mdl["dictionary"])
    g_hat = np.asarray(mdl["codes"])
    recon = d_hat @ g_hat
    out["synthetic_ksvd_relerr"] = float(np.linalg.norm(x - recon) / np.linalg.norm(x))
    if out["synthetic_ksvd_relerr"] > 0.2:
        raise ValueError(f"ksvd recon off: {out['synthetic_ksvd_relerr']}")
    return out
