"""Residues at poles via Laurent coefficient + residue theorem (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def residue_simple(f, z0: complex) -> complex:
    """Res(f, z0) for a simple pole: lim_{z->z0} (z - z0) f(z)."""
    h = 1e-6
    return complex((z0 + h - z0) * f(z0 + h))


def residue_via_circle(f, z0: complex, r: float = 1e-3, n: int = 2048) -> complex:
    """Res(f,z0) = (1/2pi i) oint_{|z-z0|=r} f dz."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = z0 + r * np.exp(1j * t)
    dz = 1j * r * np.exp(1j * t) * (2 * np.pi / n)
    return complex(np.sum(f(z) * dz) / (2j * np.pi))


def _bench_residue_calc(seed: int = 0) -> float:
    checks = []
    # f = 1/(z(z-1)): Res_0 = -1, Res_1 = +1
    f1 = lambda z: 1.0 / (z * (z - 1.0))
    checks.append(abs(residue_via_circle(f1, 0.0) + 1.0) < 1e-6)
    checks.append(abs(residue_via_circle(f1, 1.0) - 1.0) < 1e-6)
    # e^z / z^2 has Res_0 = 1 (double pole, coeff of z^1 in e^z)
    checks.append(abs(residue_via_circle(lambda z: np.exp(z) / z**2, 0.0) - 1.0) < 1e-5)
    # sin z / z^4: Res_0 = -1/6 (coeff of z^3 in sin z is -1/6)
    checks.append(
        abs(residue_via_circle(lambda z: np.sin(z) / z**4, 0.0, r=1e-2) + 1.0 / 6.0) < 1e-5
    )
    # residue theorem: sum of residues of 1/(z(z-1)) inside |z|=2 -> integral 0
    from quant_fund.models.cauchy_integral import contour_integral

    checks.append(abs(contour_integral(f1, radius=2.0)) < 1e-6)
    # simple-pole limit formula matches circle method
    checks.append(abs(residue_simple(f1, 1.0) - residue_via_circle(f1, 1.0)) < 1e-3)
    return float(sum(checks) / len(checks))


def bench_residue_calc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_residue_calc": _bench_residue_calc(seed)}
