"""Weak vs strong convergence in sequence spaces (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def weak_limit_test(vecs: np.ndarray, y: np.ndarray) -> float:
    """Max |<v_n, y>| — goes to 0 iff v_n -> 0 weakly."""
    return float(np.max(np.abs(vecs @ y)))


def strong_norms(vecs: np.ndarray) -> np.ndarray:
    return np.asarray(np.linalg.norm(vecs, axis=1))


def _bench_weak_convergence(seed: int = 0) -> float:
    checks = []
    # standard basis e_n in l2: weakly -> 0 (Bessel), strongly divergent
    n = 200
    vecs = np.eye(n)
    y = np.ones(n) / np.sqrt(n)  # in l2: sum |<e_n,y>|^2 = 1 finite
    checks.append(weak_limit_test(vecs, y) > 0)  # individual coords equal 1/sqrt(n) -> small
    checks.append(abs(weak_limit_test(vecs, y) - 1.0 / np.sqrt(n)) < 1e-9)
    checks.append(np.allclose(strong_norms(vecs), 1.0))
    # Bessel inequality: sum of squares of coords <= ||y||^2
    checks.append(float(np.sum((vecs @ y) ** 2)) <= float(y @ y) + 1e-9)
    # a genuinely vanishing sequence converges weakly to 0 within tolerance
    vanish = np.eye(n) / np.sqrt(np.arange(1, n + 1))[:, None]
    checks.append(weak_limit_test(vanish, y) < 0.08)
    # strong convergence implies weak: scaled vectors
    small = np.eye(5) * 0.01
    checks.append(weak_limit_test(small, np.ones(5)) < 0.02)
    return float(min(1.0, sum(checks) / len(checks)))


def bench_weak_convergence(seed: int = 0) -> dict[str, float]:
    return {"synthetic_weak_convergence": _bench_weak_convergence(seed)}
