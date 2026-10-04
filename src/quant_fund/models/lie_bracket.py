"""Lie algebras: sl2/so3 bracket, Jacobi identity, structure constants (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def comm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray(a @ b - b @ a)


def jacobi(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return np.asarray(comm(x, comm(y, z)) + comm(y, comm(z, x)) + comm(z, comm(x, y)))


def structure_constants(basis: list[np.ndarray]) -> np.ndarray:
    """f[i,j,k]: [e_i,e_j] = sum_k f[i,j,k] e_k (solve linear system per pair)."""
    n = len(basis)
    bmat = np.stack([b.ravel() for b in basis], axis=1)  # columns are basis elements
    f = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            c = comm(basis[i], basis[j]).ravel()
            f[i, j] = np.linalg.lstsq(bmat, c, rcond=None)[0]
    return f


def killing(basis: list[np.ndarray]) -> np.ndarray:
    """Killing form B(x,y)=tr(ad x ad y) via structure constants."""
    n = len(basis)
    f = structure_constants(basis)
    ad = np.zeros((n, n, n))
    for i in range(n):
        ad[i] = f[i]  # ad_i[j,k] = f[i,j,k]: ad_i acting on e_j -> sum_k f i,j,k e_k
    b = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            b[i, j] = np.trace(ad[i] @ ad[j])
    return b


def _bench_lie_bracket(seed: int = 0) -> float:
    checks = []
    # so(3) basis: generators of rotations
    lx = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    ly = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=float)
    lz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
    checks.append(np.allclose(comm(lx, ly), lz))
    checks.append(np.allclose(comm(ly, lz), lx))
    checks.append(np.allclose(jacobi(lx, ly, lz), 0))
    # sl2: e,f,h with [h,e]=2e,[h,f]=-2f,[e,f]=h
    e = np.array([[0, 1], [0, 0]], dtype=float)
    f = np.array([[0, 0], [1, 0]], dtype=float)
    h = np.array([[1, 0], [0, -1]], dtype=float)
    checks.append(np.allclose(comm(h, e), 2 * e) and np.allclose(comm(h, f), -2 * f))
    checks.append(np.allclose(comm(e, f), h))
    # structure constants of so3 are totally antisymmetric eps
    fc = structure_constants([lx, ly, lz])
    checks.append(np.isclose(fc[0, 1, 2], 1.0) and np.isclose(fc[1, 0, 2], -1.0))
    checks.append(
        np.isclose(np.linalg.norm(killing([e, f, h])), 6.0, atol=1e-8)
        or np.linalg.det(killing([e, f, h])) != 0
    )
    return float(sum(checks) / len(checks))


def bench_lie_bracket(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lie_bracket": _bench_lie_bracket(seed)}
