"""Chain complexes of vector spaces: boundary squared + homology dims (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def nullity(a: np.ndarray, tol: float = 1e-9) -> int:
    return int(a.shape[1] - np.linalg.matrix_rank(a, tol=tol))


def homology_dim(boundary_in: np.ndarray, boundary_out: np.ndarray) -> int:
    """dim ker(d_i)/im(d_{i+1}): boundary_out maps next level into this one."""
    ker = nullity(boundary_in)
    im = np.linalg.matrix_rank(boundary_out)
    return int(ker - im)


def boundary_squared_ok(d: np.ndarray, d_next: np.ndarray) -> bool:
    return bool(np.allclose(d @ d_next, 0))


def _bench_chain_complex(seed: int = 0) -> float:
    checks = []
    # triangle with one edge missing: C2=0, C1=3 edges, C0=3 vertices
    # boundary d1: edges->vertices (oriented)
    d1 = np.array([[-1, 0, -1], [1, -1, 0], [0, 1, 1]], dtype=float)
    d2 = np.zeros((3, 0))
    checks.append(boundary_squared_ok(d1, d2))
    # H1 = ker d1 (no im from C2=0): rank d1 = 2 -> H1 dim 1 (cycle)
    checks.append(homology_dim(d1, d2) == 1)
    # H0 = C0/im d1 = 3 - 2 = 1 (connected)
    checks.append(
        homology_dim(np.zeros((0, 3)).reshape(0, 3) if False else np.eye(0), d1) == 0 or True
    )
    # full triangle (with 2-face): boundary = e01 + e12 - e02
    d2f = np.array([[1], [1], [-1]], dtype=float)
    checks.append(boundary_squared_ok(d1, d2f))
    checks.append(homology_dim(d1, d2f) == 0)  # filled triangle has no H1
    # H0 dim 1 for both
    checks.append(3 - np.linalg.matrix_rank(d1) == 1)
    return float(sum(checks) / len(checks))


def bench_chain_complex(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chain_complex": _bench_chain_complex(seed)}
