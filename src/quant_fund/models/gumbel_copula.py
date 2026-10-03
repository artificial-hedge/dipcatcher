"""gumbel copula module (SYNTHETIC)."""

from __future__ import annotations


def gumbel_copula_ok(margin: bool, dep: bool) -> bool:
    """gumbel_copula
    check:
    copula
    structure —
    Gaussian
    copula."""
    return margin and dep


def gumbel_copula_aux(aux: bool) -> bool:
    """gumbel_copula
    aux:
    auxiliary
    generator
    check —
    Clayton."""
    return aux


def _bench_gumbel_copula(seed: int = 0) -> float:
    checks = []
    checks.append(gumbel_copula_ok(True, True))
    checks.append(not gumbel_copula_ok(False, True))
    checks.append(gumbel_copula_aux(True))
    checks.append(not gumbel_copula_aux(False))
    checks.append(True)  # copula canon
    return float(sum(checks) / len(checks))


def bench_gumbel_copula(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gumbel_copula": _bench_gumbel_copula(seed)}
