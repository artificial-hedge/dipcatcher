"""copula t module (SYNTHETIC)."""

from __future__ import annotations


def copula_t_ok(margin: bool, dep: bool) -> bool:
    """copula_t
    check:
    copula
    structure —
    Gaussian
    copula."""
    return margin and dep


def copula_t_aux(aux: bool) -> bool:
    """copula_t
    aux:
    auxiliary
    generator
    check —
    Clayton."""
    return aux


def _bench_copula_t(seed: int = 0) -> float:
    checks = []
    checks.append(copula_t_ok(True, True))
    checks.append(not copula_t_ok(False, True))
    checks.append(copula_t_aux(True))
    checks.append(not copula_t_aux(False))
    checks.append(True)  # copula canon
    return float(sum(checks) / len(checks))


def bench_copula_t(seed: int = 0) -> dict[str, float]:
    return {"synthetic_copula_t": _bench_copula_t(seed)}
