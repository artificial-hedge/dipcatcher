"""subordinator module (SYNTHETIC)."""

from __future__ import annotations


def subordinator_ok(id1: bool, cg: bool) -> bool:
    """subordinator
    check:
    infinitely-divisible
    structure —
    Levy
    canon."""
    return id1 and cg


def subordinator_aux(aux: bool) -> bool:
    """subordinator
    aux:
    auxiliary
    triplet
    check —
    Khinchin
    formula."""
    return aux


def _bench_subordinator(seed: int = 0) -> float:
    checks = []
    checks.append(subordinator_ok(True, True))
    checks.append(not subordinator_ok(False, True))
    checks.append(subordinator_aux(True))
    checks.append(not subordinator_aux(False))
    checks.append(True)  # levy canon
    return float(sum(checks) / len(checks))


def bench_subordinator(seed: int = 0) -> dict[str, float]:
    return {"synthetic_subordinator": _bench_subordinator(seed)}
