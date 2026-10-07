"""Cyclic Jacobi eigenvalue algorithm: sweeps of Givens rotations (SYNTHETIC)
zeroing off-diagonal mass — full symmetric spectrum recovery.
"""

from __future__ import annotations

import numpy as np

from quant_fund.models._eig_synth import sym_planted


def _jacobi(A: np.ndarray, sweeps: int = 8) -> np.ndarray:
    A = A.copy()
    n = A.shape[0]
    for _ in range(sweeps):
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p, q]) < 1e-12:
                    continue
                theta = (A[q, q] - A[p, p]) / (2 * A[p, q])
                t = np.sign(theta) / (abs(theta) + np.sqrt(theta**2 + 1)) if theta != 0 else 1.0
                c = 1 / np.sqrt(t**2 + 1)
                s = t * c
                for k in range(n):
                    akp, akq = A[k, p], A[k, q]
                    A[k, p] = c * akp - s * akq
                    A[k, q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p, k], A[q, k]
                    A[p, k] = c * apk - s * aqk
                    A[q, k] = s * apk + c * aqk
    return np.sort(np.diag(A))


def bench_jacobi_eig(seed: int = 3001) -> dict[str, float]:
    A, lam = sym_planted(seed)
    est = _jacobi(A)
    err = float(np.abs(est - np.sort(lam)).mean())
    offdiag = float(np.abs(A - np.diag(np.diag(A))).max())
    return {
        "synthetic_jacobi_spec_err": err,
        "synthetic_jacobi_offdiag": offdiag,
        "synthetic_torch_available": 0.0,
    }
