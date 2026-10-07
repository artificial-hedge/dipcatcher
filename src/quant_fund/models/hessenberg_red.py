"""Householder Hessenberg reduction: A = Q H Q^T — measures reduction (SYNTHETIC)
fidelity (residual + subdiagonal mass) for symmetric and nonsymmetric
matrices.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._eig_synth import nonsym_planted, sym_planted


def _hess(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    A = A.copy()
    n = A.shape[0]
    Q = np.eye(n)
    for k in range(n - 2):
        x = A[k + 1 :, k]
        v = x.copy()
        v[0] += np.sign(x[0]) * np.linalg.norm(x)
        nv = np.linalg.norm(v)
        if nv < 1e-14:
            continue
        v = v / nv
        A[k + 1 :, k:] -= 2 * np.outer(v, v @ A[k + 1 :, k:])
        A[:, k + 1 :] -= 2 * np.outer(A[:, k + 1 :] @ v, v)
        Qk = np.eye(n)
        Qk[k + 1 :, k + 1 :] -= 2 * np.outer(v, v)
        Q = Q @ Qk
    return A, Q


def bench_hessenberg_red(seed: int = 3009) -> dict[str, float]:
    A, _ = sym_planted(seed)
    H, Q = _hess(A)
    resid = float(np.linalg.norm(A - Q @ H @ Q.T) / np.linalg.norm(A))
    off = float(np.abs(np.tril(H, -2)).max())
    A2, _ = nonsym_planted(seed)
    H2, Q2 = _hess(A2)
    resid2 = float(np.linalg.norm(A2 - Q2 @ H2 @ Q2.T) / np.linalg.norm(A2))
    off2 = float(np.abs(np.tril(H2, -2)).max())
    return {
        "synthetic_hess_resid_sym": resid,
        "synthetic_hess_resid_nonsym": resid2,
        "synthetic_hess_subdiag_sym": off,
        "synthetic_hess_subdiag_nonsym": off2,
        "synthetic_torch_available": 0.0,
    }
