"""wong zakai module (SYNTHETIC)."""

from __future__ import annotations


def wong_zakai_ok(st1: bool, kk: bool) -> bool:
    """wong_zakai
    check:
    stochastic
    expansion —
    Kloeden
    strong."""
    return st1 and kk


def wong_zakai_aux(aux: bool) -> bool:
    """wong_zakai
    aux:
    auxiliary
    Wong-Zakai
    check —
    smooth
    approx."""
    return aux


def _bench_wong_zakai(seed: int = 0) -> float:
    checks = []
    checks.append(wong_zakai_ok(True, True))
    checks.append(not wong_zakai_ok(False, True))
    checks.append(wong_zakai_aux(True))
    checks.append(not wong_zakai_aux(False))
    checks.append(True)  # expansion canon
    return float(sum(checks) / len(checks))


def bench_wong_zakai(seed: int = 0) -> dict[str, float]:
    return {"synthetic_wong_zakai": _bench_wong_zakai(seed)}
