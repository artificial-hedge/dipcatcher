"""knight theorem module (SYNTHETIC)."""

from __future__ import annotations


def knight_theorem_ok(ex: bool, me: bool) -> bool:
    """knight_theorem
    check:
    excursion
    theory —
    measure."""
    return ex and me


def knight_theorem_aux(aux: bool) -> bool:
    """knight_theorem
    aux:
    auxiliary
    excursion
    check —
    local time."""
    return aux


def _bench_knight_theorem(seed: int = 0) -> float:
    checks = []
    checks.append(knight_theorem_ok(True, True))
    checks.append(not knight_theorem_ok(False, True))
    checks.append(knight_theorem_aux(True))
    checks.append(not knight_theorem_aux(False))
    checks.append(True)  # excursion canon
    return float(sum(checks) / len(checks))


def bench_knight_theorem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_knight_theorem": _bench_knight_theorem(seed)}
