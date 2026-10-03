"""Resolvent (A - z I)^{-1}: Neumann series + distance-to-spectrum bound (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def resolvent(a: np.ndarray, z: float) -> np.ndarray:
    return np.asarray(np.linalg.inv(a - z * np.eye(a.shape[0])))


def _bench_compact_resolvent(seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    checks = []
    m = rng.normal(size=(5, 5))
    a = (m + m.T) / 4  # self-adjoint, norm < ~1.2
    # resolvent identity: (A - zI) R = I
    z = 5.0
    r = resolvent(a, z)
    checks.append(np.allclose((a - z * np.eye(5)) @ r, np.eye(5)))
    # Neumann series for |z| > ||A||: R(z) = -sum A^k / z^{k+1}
    series = np.zeros((5, 5))
    for k in range(60):
        series += np.linalg.matrix_power(a, k) / z ** (k + 1)
    checks.append(np.allclose(r, -series, atol=1e-9))
    # bound: ||R|| <= 1 / dist(z, sigma(A)) for self-adjoint (equality)
    dist = float(np.min(np.abs(np.linalg.eigvalsh(a) - z)))
    checks.append(abs(np.linalg.norm(r, 2) - 1.0 / dist) < 1e-9)
    # resolvent commutes: R(z1) R(z2) = R(z2) R(z1)
    r2 = resolvent(a, 7.0)
    checks.append(np.allclose(r @ r2, r2 @ r))
    # resolvent equation: R(z1) - R(z2) = (z1 - z2) R1 R2
    checks.append(np.allclose(r - r2, (5.0 - 7.0) * r @ r2))
    return float(sum(checks) / len(checks))


def bench_compact_resolvent(seed: int = 0) -> dict[str, float]:
    return {"synthetic_compact_resolvent": _bench_compact_resolvent(seed)}
