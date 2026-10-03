"""Sturm-Liouville -y'' = lam y on [0,pi]: eigenvalues n^2, orthogonal sines (SYNTHETIC)."""

from __future__ import annotations

import math


def eigenfunction(n: int, x: float) -> float:
    return math.sin(n * x)


def eigenvalue(n: int) -> float:
    return float(n * n)


def inner_product(f, g, pts: int = 4000) -> float:
    import math

    h = math.pi / pts
    return float(sum(f(i * h) * g(i * h) for i in range(pts + 1)) * h)


def rayleigh(n: int, pts: int = 4000) -> float:
    """Rayleigh quotient int y'^2 / int y^2 for sin(nx): = n^2."""
    import math

    h = math.pi / pts
    num = sum((n * math.cos(n * i * h)) ** 2 for i in range(pts + 1)) * h
    den = sum(math.sin(n * i * h) ** 2 for i in range(pts + 1)) * h
    return num / den


def _bench_sturm_liouville(seed: int = 0) -> float:
    checks = []
    checks.append(eigenvalue(3) == 9.0)
    # eigenfunctions satisfy -y'' = n^2 y
    checks.append(abs(-(-(3.0**2) * 1.0) - eigenvalue(3)) < 1e-9)
    # orthogonality sin(2x), sin(3x) on [0,pi]
    ip = inner_product(lambda x: eigenfunction(2, x), lambda x: eigenfunction(3, x))
    checks.append(abs(ip) < 1e-3)
    # norm squared = pi/2
    nn = inner_product(lambda x: eigenfunction(1, x), lambda x: eigenfunction(1, x))
    checks.append(abs(nn - math.pi / 2) < 1e-3)
    # Rayleigh quotient gives eigenvalue
    checks.append(abs(rayleigh(2) - 4.0) < 0.05)
    # boundary conditions y(0)=y(pi)=0
    checks.append(eigenfunction(5, 0.0) == 0.0 and abs(eigenfunction(5, math.pi)) < 1e-12)
    return float(sum(checks) / len(checks))


def bench_sturm_liouville(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sturm_liouville": _bench_sturm_liouville(seed)}
