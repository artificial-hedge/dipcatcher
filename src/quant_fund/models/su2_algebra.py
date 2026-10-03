"""su(2): skew-Hermitian traceless 2x2, Pauli algebra, exp to SU(2) (SYNTHETIC)."""

from __future__ import annotations

import numpy as np

S1 = -0.5j * np.array([[0.0, 1.0], [1.0, 0.0]])  # sigma_x/(2i)
S2 = -0.5j * np.array([[0.0, -1.0j], [1.0j, 0.0]])  # sigma_y/(2i)
S3 = -0.5j * np.array([[1.0, 0.0], [0.0, -1.0]])  # sigma_z/(2i)


def brac(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.asarray(x @ y - y @ x)


def is_su2_elem(x: np.ndarray, tol: float = 1e-9) -> bool:
    return bool(np.allclose(x + x.conj().T, 0, atol=tol) and abs(np.trace(x)) < tol)


def _bench_su2_algebra(seed: int = 0) -> float:
    checks = []
    # basis is skew-Hermitian traceless
    checks.append(all(is_su2_elem(s) for s in (S1, S2, S3)))
    # [S1,S2]=S3 cyclic
    checks.append(np.allclose(brac(S1, S2), S3))
    checks.append(np.allclose(brac(S2, S3), S1))
    checks.append(np.allclose(brac(S3, S1), S2))
    # closure: brackets of combos stay in su2
    x = 0.3 * S1 + 0.5 * S2 + 0.7 * S3
    y = 1.1 * S1 - 0.4 * S2 + 0.2 * S3
    checks.append(is_su2_elem(brac(x, y)))
    # exp maps to SU(2): U unitary det 1
    u = np.asarray(_expm(x))
    checks.append(np.allclose(u @ u.conj().T, np.eye(2), atol=1e-8))
    checks.append(np.isclose(np.linalg.det(u), 1.0, atol=1e-8))
    return float(sum(checks) / len(checks))


def _expm(a: np.ndarray, terms: int = 30) -> np.ndarray:
    out = np.eye(2, dtype=complex)
    term = np.eye(2, dtype=complex)
    for k in range(1, terms):
        term = term @ a / k
        out = out + term
    return out


def bench_su2_algebra(seed: int = 0) -> dict[str, float]:
    return {"synthetic_su2_algebra": _bench_su2_algebra(seed)}
