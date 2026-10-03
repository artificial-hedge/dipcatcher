"""Contour integration over a circle: Cauchy's theorem and 2*pi*i residues (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def contour_integral(f, radius: float = 1.0, center: complex = 0j, n: int = 4096) -> complex:
    """Integral of f over the positively oriented circle |z-c|=r."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = center + radius * np.exp(1j * t)
    dz = 1j * radius * np.exp(1j * t) * (2 * np.pi / n)
    return complex(np.sum(f(z) * dz))


def _bench_cauchy_integral(seed: int = 0) -> float:
    checks = []
    # (1/2pi i) oint dz/z = 1 on |z|=1
    val = contour_integral(lambda z: 1.0 / z)
    checks.append(abs(val / (2j * np.pi) - 1.0) < 1e-10)
    # oint dz/z^2 = 0 (residue of z^-2 is 0)
    checks.append(abs(contour_integral(lambda z: 1.0 / z**2)) < 1e-9)
    # oint z dz = 0 (holomorphic)
    checks.append(abs(contour_integral(lambda z: z)) < 1e-9)
    # oint conj(z) dz = 2 pi i on |z|=1 (z-bar = 1/z on circle, not holomorphic)
    checks.append(abs(contour_integral(lambda z: np.conj(z)) - 2j * np.pi) < 1e-9)
    # Cauchy integral formula: (1/2pi i) oint e^z/(z) dz = e^0 = 1
    checks.append(abs(contour_integral(lambda z: np.exp(z) / z) / (2j * np.pi) - 1.0) < 1e-10)
    # off-center: oint dz/(z - a) = 2pi i iff |a| < r
    checks.append(abs(contour_integral(lambda z: 1.0 / (z - 0.3), radius=1.0) - 2j * np.pi) < 1e-9)
    checks.append(abs(contour_integral(lambda z: 1.0 / (z - 3.0), radius=1.0)) < 1e-9)
    return float(sum(checks) / len(checks))


def bench_cauchy_integral(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cauchy_integral": _bench_cauchy_integral(seed)}
