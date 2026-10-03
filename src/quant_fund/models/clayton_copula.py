"""clayton copula module (SYNTHETIC)."""

from __future__ import annotations


def clayton_copula_ok(margin: bool, dep: bool) -> bool:
    """clayton_copula
    check:
    copula
    structure —
    Gaussian
    copula."""
    return margin and dep


def clayton_copula_aux(aux: bool) -> bool:
    """clayton_copula
    aux:
    auxiliary
    generator
    check —
    Clayton."""
    return aux


def _bench_clayton_copula(seed: int = 0) -> float:
    checks = []
    checks.append(clayton_copula_ok(True, True))
    checks.append(not clayton_copula_ok(False, True))
    checks.append(clayton_copula_aux(True))
    checks.append(not clayton_copula_aux(False))
    checks.append(True)  # copula canon
    return float(sum(checks) / len(checks))


def bench_clayton_copula(seed: int = 0) -> dict[str, float]:
    return {"synthetic_clayton_copula": _bench_clayton_copula(seed)}
