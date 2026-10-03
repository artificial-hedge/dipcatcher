"""SYNTHETIC Sturm-sequence bisection eigenvalue finder.

Count eigenvalues below a shift via Sturm sequence of a symmetric
tridiagonal (from Householder reduction); bisection isolates each
eigenvalue — verified against eigh.
"""

from __future__ import annotations

import random

import numpy as np


def _tridiag(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Householder tridiagonalization: returns (diag, offdiag)."""
    n = A.shape[0]
    T = A.astype(float).copy()
    for k in range(n - 2):
        x = T[k + 1 :, k]
        e = np.zeros_like(x)
        e[0] = -np.sign(x[0]) * np.linalg.norm(x) if x[0] != 0 else -np.linalg.norm(x)
        v = x - e
        nv = np.linalg.norm(v)
        if nv < 1e-14:
            continue
        v /= nv
        T[k + 1 :, k:] -= 2 * np.outer(v, v @ T[k + 1 :, k:])
        T[:, k + 1 :] -= 2 * np.outer(T[:, k + 1 :] @ v, v)
    return np.diag(T).copy(), np.diag(T, 1).copy()


def _sturm_count(d: np.ndarray, e: np.ndarray, x: float) -> int:
    """Number of eigenvalues < x for tridiag(d, e)."""
    n = len(d)
    count = 0
    q = d[0] - x
    if q <= 0:
        count += 1
    for i in range(1, n):
        if q == 0:
            q = 1e-14
        q = d[i] - x - e[i - 1] ** 2 / q
        if q <= 0:
            count += 1
    return count


def sturm_eigs(A: np.ndarray) -> np.ndarray:
    """All eigenvalues of symmetric A via Sturm bisection."""
    d, e = _tridiag(A)
    n = len(d)
    lo = float(np.min(d) - 2 * np.linalg.norm(e) - 1)
    hi = float(np.max(d) + 2 * np.linalg.norm(e) + 1)
    out = []
    for k in range(n):
        a, b = lo, hi
        for _ in range(60):
            mid = (a + b) / 2
            if _sturm_count(d, e, mid) > k:
                b = mid
            else:
                a = mid
        out.append((a + b) / 2)
    return np.array(out)


def bench_sturm_eig(seed: int = 20261231 + 535) -> dict[str, float]:
    rng = random.Random(seed)
    eig_ok = 0
    cnt_ok = 0
    n_trials = 25
    for _ in range(n_trials):
        n = rng.randrange(4, 8)
        M = np.array([[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)])
        A = M + M.T
        got = sturm_eigs(A)
        ref = np.linalg.eigvalsh(A)
        eig_ok += int(np.allclose(np.sort(got), ref, atol=1e-5))
        # count invariant vs shifted Sturm count
        d, e = _tridiag(A)
        x = (ref[0] + ref[-1]) / 2
        cnt_ok += int(_sturm_count(d, e, x) == int(np.sum(ref < x)))
    return {
        "synthetic_eigs_exact": eig_ok / n_trials,
        "synthetic_sturm_count": cnt_ok / n_trials,
    }
