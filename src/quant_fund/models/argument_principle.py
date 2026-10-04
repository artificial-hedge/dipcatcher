"""Argument principle: winding of f(gamma) counts zeros minus poles (SYNTHETIC)."""

from __future__ import annotations

import numpy as np


def winding(f, radius: float, center: complex = 0j, n: int = 8192) -> float:
    """(1/2pi) * change in arg f along the circle |z - c| = r."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = center + radius * np.exp(1j * t)
    fz = np.asarray(f(z), dtype=np.complex128)
    fc = np.concatenate([fz, fz[:1]])
    darg = np.angle(fc[1:] / fc[:-1])
    return float(np.sum(darg) / (2 * np.pi))


def _bench_argument_principle(seed: int = 0) -> float:
    checks = []
    # z^3 - 1 has 3 zeros inside |z| = 2
    checks.append(abs(winding(lambda z: z**3 - 1.0, 2.0) - 3.0) < 1e-6)
    # z^3 - 1 has 0 zeros inside |z| = 0.5
    checks.append(abs(winding(lambda z: z**3 - 1.0, 0.5)) < 1e-6)
    # (z^2 + 1)/z has 2 zeros and 1 pole at 0 -> winding 1 on |z|=2
    checks.append(abs(winding(lambda z: (z**2 + 1.0) / z, 2.0) - 1.0) < 1e-6)
    # e^z has no zeros anywhere -> winding 0
    checks.append(abs(winding(lambda z: np.exp(z), 3.0)) < 1e-6)
    # multiplicity counts: z^2 -> 2 on unit circle
    checks.append(abs(winding(lambda z: z**2, 1.0) - 2.0) < 1e-6)
    # Rouche check: z^5 + z^2 - 1 has all 5 roots in |z| < 1.5 (|z^5|=7.6 > |z^2 - 1| <= 3.25)
    checks.append(abs(winding(lambda z: z**5 + z**2 - 1.0, 1.5) - 5.0) < 1e-6)
    return float(sum(checks) / len(checks))


def bench_argument_principle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_argument_principle": _bench_argument_principle(seed)}
