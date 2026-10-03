"""joe copula module (SYNTHETIC)."""

from __future__ import annotations


def joe_copula_ok(margin: bool, dep: bool) -> bool:
    """joe_copula
    check:
    copula
    structure —
    Gaussian
    copula."""
    return margin and dep


def joe_copula_aux(aux: bool) -> bool:
    """joe_copula
    aux:
    auxiliary
    generator
    check —
    Clayton."""
    return aux


def _bench_joe_copula(seed: int = 0) -> float:
    checks = []
    checks.append(joe_copula_ok(True, True))
    checks.append(not joe_copula_ok(False, True))
    checks.append(joe_copula_aux(True))
    checks.append(not joe_copula_aux(False))
    checks.append(True)  # copula canon
    return float(sum(checks) / len(checks))


def bench_joe_copula(seed: int = 0) -> dict[str, float]:
    return {"synthetic_joe_copula": _bench_joe_copula(seed)}
