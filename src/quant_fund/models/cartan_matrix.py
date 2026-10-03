"""Cartan matrices of simple Lie algebras A2, B2, G2 (SYNTHETIC)."""

from __future__ import annotations

import numpy as np

A2 = np.array([[2, -1], [-1, 2]])
B2 = np.array([[2, -2], [-1, 2]])
G2 = np.array([[2, -1], [-3, 2]])


def is_cartan(c: np.ndarray) -> bool:
    """Cartan axioms: diag=2, off-diag in {0,-1,-2,-3}, symmetrizable."""
    n = c.shape[0]
    if not all(c[i, i] == 2 for i in range(n)):
        return False
    for i in range(n):
        for j in range(n):
            if i != j and c[i, j] not in (0, -1, -2, -3):
                return False
    # symmetrizable: C[i,j]!=0 iff C[j,i]!=0
    for i in range(n):
        for j in range(n):
            if (c[i, j] != 0) != (c[j, i] != 0):
                return False
    return bool(np.linalg.det(c) > 0)


def _bench_cartan_matrix(seed: int = 0) -> float:
    checks = []
    checks.append(is_cartan(A2))
    checks.append(is_cartan(B2))
    checks.append(is_cartan(G2))
    checks.append(np.isclose(np.linalg.det(A2), 3.0))
    checks.append(np.isclose(np.linalg.det(B2), 2.0))
    checks.append(np.isclose(np.linalg.det(G2), 1.0))
    # A2 symmetric; B2/G2 not
    checks.append(np.array_equal(A2, A2.T) and not np.array_equal(B2, B2.T))
    # negative test: non-Cartan
    checks.append(not is_cartan(np.array([[2, -4], [-1, 2]])))
    return float(sum(checks) / len(checks))


def bench_cartan_matrix(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cartan_matrix": _bench_cartan_matrix(seed)}
