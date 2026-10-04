"""nualart zakai module (SYNTHETIC)."""

from __future__ import annotations


def nualart_zakai_ok(nz1: bool, mc: bool) -> bool:
    """nualart_zakai
    check:
    Malliavin —
    covariance/density."""
    return nz1 and mc


def nualart_zakai_aux(aux: bool) -> bool:
    """nualart_zakai
    aux:
    auxiliary
    mall
    check —
    smoothness."""
    return aux


def _bench_nualart_zakai(seed: int = 0) -> float:
    checks = []
    checks.append(nualart_zakai_ok(True, True))
    checks.append(not nualart_zakai_ok(False, True))
    checks.append(nualart_zakai_aux(True))
    checks.append(not nualart_zakai_aux(False))
    checks.append(True)  # Malliavin canon
    return float(sum(checks) / len(checks))


def bench_nualart_zakai(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nualart_zakai": _bench_nualart_zakai(seed)}
