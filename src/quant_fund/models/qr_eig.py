"""Unshifted + Wilkinson-shifted QR algorithm on a Hessenberg form —
full nonsymmetric-spectrum recovery vs planted eigenvalues.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._eig_synth import nonsym_planted


def _hessenberg(A: np.ndarray) -> np.ndarray:
    A = A.copy()
    n = A.shape[0]
    for k in range(n - 2):
        x = A[k + 1 :, k]
        v = x.copy()
        v[0] += np.sign(x[0]) * np.linalg.norm(x)
        v = v / np.linalg.norm(v)
        A[k + 1 :, k:] -= 2 * np.outer(v, v @ A[k + 1 :, k:])
        A[:, k + 1 :] -= 2 * np.outer(A[:, k + 1 :] @ v, v)
    return A


def _qr_eigs(H: np.ndarray, iters: int = 300) -> np.ndarray:
    H = H.copy()
    n = H.shape[0]
    for _ in range(iters):
        mu = H[n - 1, n - 1]  # Rayleigh shift
        Q, R = np.linalg.qr(H - mu * np.eye(n))
        H = R @ Q + mu * np.eye(n)
        if abs(H[n - 1, n - 2]) < 1e-10 * (abs(H[n - 1, n - 1]) + abs(H[n - 2, n - 2])):
            if n > 2:
                return np.concatenate([_qr_eigs(H[: n - 1, : n - 1], iters), [H[n - 1, n - 1]]])
            break
    return np.diag(H)


def bench_qr_eig(seed: int = 3005) -> dict[str, float]:
    A, lam = nonsym_planted(seed)
    H = _hessenberg(A)
    est = _qr_eigs(H)
    est_sorted = np.sort(np.real(est))[::-1]
    err = float(np.abs(est_sorted[: len(lam)] - np.sort(lam)[::-1]).mean())
    return {
        "synthetic_qr_spec_err": err,
        "synthetic_hess_offdiag": float(np.abs(np.tril(H, -2)).max()),
        "torch_available": 0.0,
    }
