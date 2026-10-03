"""Spectral theorem for symmetric matrices: A = Q Lambda Q^T (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def spectral_decomp(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Q, w) with A = Q diag(w) Q^T."""
    w, q = np.linalg.eigh(a)
    return q, w


def rayleigh_max(a: np.ndarray, iters: int = 200) -> float:
    x = np.ones(a.shape[0])
    for _ in range(iters):
        y = a @ x
        x = y / np.linalg.norm(y)
    return float(x @ a @ x / (x @ x))


def _bench_spectral_theorem(seed: int = 0) -> float:
    checks = []
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((4, 4))
    a = m + m.T  # symmetric
    q, w = spectral_decomp(a)
    checks.append(np.allclose(q @ q.T, np.eye(4)))
    checks.append(np.allclose(a, q @ np.diag(w) @ q.T))
    # Rayleigh quotient approaches the dominant eigenvalue (largest |lambda|)
    checks.append(abs(rayleigh_max(a) - w[np.argmax(np.abs(w))]) < 0.02)
    # eigendecomp of projection matrix has eigenvalues 0/1
    v = np.array([1.0, 0.0, 0.0, 0.0])
    p = np.outer(v, v)
    _, wp = spectral_decomp(p)
    checks.append(np.allclose(sorted(wp), [0.0, 0.0, 0.0, 1.0]))
    # functional calculus: A^2 via spectral
    checks.append(np.allclose(a @ a, q @ np.diag(w**2) @ q.T))
    return float(sum(checks) / len(checks))


def bench_spectral_theorem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_theorem": _bench_spectral_theorem(seed)}
