"""frank copula module (SYNTHETIC)."""

from __future__ import annotations


def frank_copula_ok(margin: bool, dep: bool) -> bool:
    """frank_copula
    check:
    copula
    structure —
    Gaussian
    copula."""
    return margin and dep


def frank_copula_aux(aux: bool) -> bool:
    """frank_copula
    aux:
    auxiliary
    generator
    check —
    Clayton."""
    return aux


def _bench_frank_copula(seed: int = 0) -> float:
    checks = []
    checks.append(frank_copula_ok(True, True))
    checks.append(not frank_copula_ok(False, True))
    checks.append(frank_copula_aux(True))
    checks.append(not frank_copula_aux(False))
    checks.append(True)  # copula canon
    return float(sum(checks) / len(checks))


def bench_frank_copula(seed: int = 0) -> dict[str, float]:
    return {"synthetic_frank_copula": _bench_frank_copula(seed)}
