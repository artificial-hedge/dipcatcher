"""Cauchy estimates -> Liouville; maximum modulus principle (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def cauchy_derivative(f, z0: complex, r: float, n: int = 4096) -> complex:
    """f'(z0) = (1/2pi i) oint f(z)/(z - z0)^2 dz over |z - z0| = r."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = z0 + r * np.exp(1j * t)
    dz = 1j * r * np.exp(1j * t) * (2 * np.pi / n)
    return complex(np.sum(f(z) / (z - z0) ** 2 * dz) / (2j * np.pi))


def _bench_liouville(seed: int = 0) -> float:
    checks = []
    # derivative formula recovers exact derivative: f = z^3, f'(0.5) = 0.75
    checks.append(abs(cauchy_derivative(lambda z: z**3, 0.5, 0.5) - 0.75) < 1e-8)
    # Cauchy estimate |f'(z0)| <= M(r)/r for bounded f: f = sin z, |sin| <= cosh on circle
    for r in (0.5, 1.0, 2.0):
        bound = np.cosh(r) / r
        checks.append(abs(cauchy_derivative(lambda z: np.sin(z), 0.0, r)) <= bound + 1e-9)
    # Liouville heuristic: a constant function has derivative 0 everywhere
    checks.append(abs(cauchy_derivative(lambda z: np.full_like(z, 2.5), 0.3, 1.0)) < 1e-9)

    # maximum modulus: max |z^3 - 2z| on |z|<=1 occurs on boundary, interior less
    def f(z: np.ndarray) -> np.ndarray:
        return np.asarray(z**3 - 2 * z)

    th = np.linspace(0, 2 * np.pi, 2000)
    m_b = float(np.max(np.abs(f(np.exp(1j * th)))))
    checks.append(m_b > abs(f(0.5)))
    # interior point cannot exceed boundary max for holomorphic non-constant f
    rng = np.random.default_rng(seed)
    interior = rng.uniform(-0.9, 0.9, 400) + 1j * rng.uniform(-0.9, 0.9, 400)
    interior = interior[np.abs(interior) <= 1.0]
    checks.append(bool(np.all(np.abs(f(interior)) <= m_b + 1e-12)))
    return float(sum(checks) / len(checks))


def bench_liouville(seed: int = 0) -> dict[str, float]:
    return {"synthetic_liouville": _bench_liouville(seed)}
