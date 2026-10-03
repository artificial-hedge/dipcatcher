"""Schur orthogonality relations on S3 irreps (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def std_rep(p: tuple[int, ...]) -> np.ndarray:
    """Standard 2-dim rep of S3 on {(x,y): x+y+z=0} as perm matrices minus triv.
    Use: std(g) = P_g restricted; compute via basis e1-e3, e2-e3."""
    n = len(p)
    P = np.zeros((n, n))
    for i in range(n):
        P[i, p[i]] = 1.0
    # invariant subspace basis: v1 = e0-e2, v2 = e1-e2
    B = np.array([[1, 0, -1], [0, 1, -1]], dtype=float).T  # 3x2
    coords = np.linalg.lstsq(B, P @ B, rcond=None)[0]
    return np.asarray(coords)


def inner_prod_rows(
    chi1: dict[int, float], chi2: dict[int, float], sizes: dict[int, int], order: int
) -> float:
    return sum(chi1[c] * chi2[c] * sizes[c] for c in sizes) / order


def _bench_schur_ortho(seed: int = 0) -> float:
    checks = []
    e = (0, 1, 2)
    t = (1, 0, 2)
    c = (1, 2, 0)
    m_e = std_rep(e)
    m_t = std_rep(t)
    m_c = std_rep(c)
    checks.append(np.allclose(m_e, np.eye(2)))
    checks.append(np.allclose(m_t @ m_t, np.eye(2)))
    checks.append(np.allclose(m_c @ m_c @ m_c, np.eye(2), atol=1e-8))
    # character of std rep: tr
    checks.append(np.isclose(np.trace(m_e), 2.0))
    checks.append(np.isclose(np.trace(m_t), 0.0))
    checks.append(np.isclose(np.trace(m_c), -1.0))
    return float(sum(checks) / len(checks))


def bench_schur_ortho(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schur_ortho": _bench_schur_ortho(seed)}
