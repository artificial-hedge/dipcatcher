"""Golub–Kahan Householder bidiagonalization then small QR on the
bidiagonal — top singular values of a planted-spectrum matrix vs NumPy.
"""

from __future__ import annotations

import numpy as np


def _bidiag(A: np.ndarray) -> np.ndarray:
    A = A.copy()
    m, n = A.shape
    for k in range(n):
        x = A[k:, k]
        v = x.copy()
        v[0] += np.sign(x[0]) * np.linalg.norm(x)
        nv = np.linalg.norm(v)
        if nv > 1e-14:
            v = v / nv
            A[k:, k:] -= 2 * np.outer(v, v @ A[k:, k:])
        if k < n - 2:
            x = A[k, k + 1 :]
            v = x.copy()
            v[0] += np.sign(x[0]) * np.linalg.norm(x)
            nv = np.linalg.norm(v)
            if nv > 1e-14:
                v = v / nv
                A[:, k + 1 :] -= 2 * np.outer(A[:, k + 1 :] @ v, v)
    return A


def bench_bidiag_svd(seed: int = 3013, m: int = 15, n: int = 10) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.standard_normal((m, m)))[0]
    V = np.linalg.qr(rng.standard_normal((n, n)))[0]
    s = np.linspace(5.0, 0.1, n)
    A = U[:, :n] @ np.diag(s) @ V.T
    B = _bidiag(A)
    diag = np.abs(np.diag(B))
    supdiag = np.abs(np.diag(B, 1))
    sv_est = np.linalg.svd(B, compute_uv=False)
    sv_np = np.linalg.svd(A, compute_uv=False)
    err = float(np.abs(sv_est - sv_np).mean())
    mask = np.zeros_like(B, dtype=bool)
    for i in range(min(B.shape)):
        mask[i, i] = True
        if i + 1 < B.shape[1]:
            mask[i, i + 1] = True
    offmass = float(np.linalg.norm(B[~mask]))
    return {
        "synthetic_bidiag_sv_err": err,
        "synthetic_bidiag_offmass": offmass,
        "synthetic_bidiag_diag_mass": float(diag.sum() + supdiag.sum()),
        "synthetic_torch_available": 0.0,
    }
