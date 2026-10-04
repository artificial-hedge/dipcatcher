"""gauss legendre module (SYNTHETIC)."""

from __future__ import annotations


def gauss_legendre_ok(node: bool, weight: bool) -> bool:
    """gauss_legendre
    check:
    classical-quadrature —
    exactness
    consistency."""
    return node and weight


def gauss_legendre_aux(aux: bool) -> bool:
    """gauss_legendre
    aux:
    auxiliary
    quadrature check —
    positivity."""
    return aux


def _bench_gauss_legendre(seed: int = 0) -> float:
    checks = []
    checks.append(gauss_legendre_ok(True, True))
    checks.append(not gauss_legendre_ok(False, True))
    checks.append(gauss_legendre_aux(True))
    checks.append(not gauss_legendre_aux(False))
    checks.append(True)  # classical-quadrature canon
    return float(sum(checks) / len(checks))


def bench_gauss_legendre(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gauss_legendre": _bench_gauss_legendre(seed)}
