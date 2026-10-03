"""SYNTHETIC one-sided Jacobi SVD.

Hestenes one-sided Jacobi rotations orthogonalize columns of A; singular
values verified against numpy.linalg.svd.
"""

from __future__ import annotations

import random

import numpy as np


def jacobi_svd(A: np.ndarray, sweeps: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-sided Jacobi: A → U Σ Vᵀ via column orthogonalization."""
    m, n = A.shape
    U = A.astype(float).copy()
    V = np.eye(n)
    for _ in range(sweeps):
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha = np.sum(U[:, p] * U[:, p])
                beta = np.sum(U[:, q] * U[:, q])
                gamma = np.sum(U[:, p] * U[:, q])
                off += abs(gamma)
                if abs(gamma) < 1e-14 * np.sqrt(alpha * beta):
                    continue
                zeta = (beta - alpha) / (2 * gamma)
                t = np.sign(zeta) / (abs(zeta) + np.sqrt(1 + zeta * zeta))
                c = 1 / np.sqrt(1 + t * t)
                s = c * t
                up = U[:, p] * c - U[:, q] * s
                uq = U[:, p] * s + U[:, q] * c
                U[:, p], U[:, q] = up, uq
                vp = V[:, p] * c - V[:, q] * s
                vq = V[:, p] * s + V[:, q] * c
                V[:, p], V[:, q] = vp, vq
        if off < 1e-12:
            break
    svals = np.linalg.norm(U, axis=0)
    order = np.argsort(-svals)
    svals = svals[order]
    U = U[:, order] / np.where(svals > 1e-14, svals, 1.0)
    V = V[:, order]
    return U, svals, V.T


def bench_jacobi_svd(seed: int = 20261231 + 532) -> dict[str, float]:
    rng = random.Random(seed)
    sv_ok = 0
    rec_ok = 0
    orth_ok = 0
    n_trials = 30
    for _ in range(n_trials):
        m = rng.randrange(4, 8)
        n = rng.randrange(2, m + 1)
        A = np.array([[rng.uniform(-2, 2) for _ in range(n)] for _ in range(m)])
        U, s, Vt = jacobi_svd(A)
        ref = np.linalg.svd(A, compute_uv=False)
        sv_ok += int(np.allclose(s, ref[: len(s)], atol=1e-6))
        rec_ok += int(np.allclose(U @ np.diag(s) @ Vt, A, atol=1e-6))
        orth_ok += int(np.allclose(Vt @ Vt.T, np.eye(n), atol=1e-8))
    return {
        "synthetic_singular_values": sv_ok / n_trials,
        "synthetic_reconstruction": rec_ok / n_trials,
        "synthetic_v_orthonormal": orth_ok / n_trials,
    }
