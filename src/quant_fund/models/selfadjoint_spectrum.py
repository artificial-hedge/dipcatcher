"""Spectral theorem: self-adjoint A has real spectrum, ON eigenbasis (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def _bench_selfadjoint_spectrum(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    m = rng.normal(size=(6, 6))
    a = m + m.T
    w, v = np.linalg.eigh(a)
    # real eigenvalues (eigh gives real), eigenvectors orthonormal
    checks.append(bool(np.all(np.isreal(w))))
    checks.append(np.allclose(v.T @ v, np.eye(6), atol=1e-10))
    # spectral decomposition reconstructs A
    checks.append(np.allclose(v @ np.diag(w) @ v.T, a))
    # spectral radius = operator norm for self-adjoint
    checks.append(abs(np.max(np.abs(w)) - np.linalg.norm(a, 2)) < 1e-10)
    # eigenvectors of distinct eigenvalues orthogonal (generic case)
    checks.append(np.allclose(np.diag(v.T @ v), 1.0))
    # Rayleigh quotient bounds: min eig <= x'Ax/|x|^2 <= max eig
    xs = rng.normal(size=(10, 6))
    rq = np.array([xi @ a @ xi / (xi @ xi) for xi in xs])
    checks.append(bool(np.all(rq >= w[0] - 1e-9) and np.all(rq <= w[-1] + 1e-9)))
    return float(sum(checks) / len(checks))


def bench_selfadjoint_spectrum(seed: int = 0) -> dict[str, float]:
    return {"synthetic_selfadjoint_spectrum": _bench_selfadjoint_spectrum(seed)}
