"""Gram-Schmidt orthonormalization + Cauchy-Schwarz/triangle inequalities (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def gram_schmidt(vectors: list[np.ndarray]) -> list[np.ndarray]:
    basis: list[np.ndarray] = []
    for v in vectors:
        w = v.copy()
        for b in basis:
            w = w - (w @ b) * b
        n = np.linalg.norm(w)
        if n > 1e-12:
            basis.append(w / n)
    return basis


def cauchy_schwarz(x: np.ndarray, y: np.ndarray) -> bool:
    return bool(abs(x @ y) <= np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)


def triangle_ineq(x: np.ndarray, y: np.ndarray) -> bool:
    return bool(np.linalg.norm(x + y) <= np.linalg.norm(x) + np.linalg.norm(y) + 1e-12)


def _bench_gram_schmidt(seed: int = 0) -> float:
    checks = []
    rng = np.random.default_rng(seed)
    vecs = [rng.standard_normal(4) for _ in range(3)]
    basis = gram_schmidt(vecs)
    checks.append(len(basis) == 3)
    checks.append(all(abs(np.linalg.norm(b) - 1.0) < 1e-9 for b in basis))
    checks.append(all(abs(basis[i] @ basis[j]) < 1e-9 for i in range(3) for j in range(i)))
    # span preserved: each original vec is combo of basis
    m = np.stack(basis)
    for v in vecs:
        coeffs = m @ v
        checks.append(np.allclose(coeffs @ m, v))
    x = rng.standard_normal(4)
    y = rng.standard_normal(4)
    checks.append(cauchy_schwarz(x, y))
    checks.append(triangle_ineq(x, y))
    return float(sum(checks) / len(checks))


def bench_gram_schmidt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gram_schmidt": _bench_gram_schmidt(seed)}
