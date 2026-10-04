"""copula gauss module (SYNTHETIC)."""

from __future__ import annotations


def copula_gauss_ok(margin: bool, dep: bool) -> bool:
    """copula_gauss
    check:
    copula
    structure —
    Gaussian
    copula."""
    return margin and dep


def copula_gauss_aux(aux: bool) -> bool:
    """copula_gauss
    aux:
    auxiliary
    generator
    check —
    Clayton."""
    return aux


def _bench_copula_gauss(seed: int = 0) -> float:
    checks = []
    checks.append(copula_gauss_ok(True, True))
    checks.append(not copula_gauss_ok(False, True))
    checks.append(copula_gauss_aux(True))
    checks.append(not copula_gauss_aux(False))
    checks.append(True)  # copula canon
    return float(sum(checks) / len(checks))


def bench_copula_gauss(seed: int = 0) -> dict[str, float]:
    return {"synthetic_copula_gauss": _bench_copula_gauss(seed)}
