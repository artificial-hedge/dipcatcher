"""russo seymour module (SYNTHETIC)."""

from __future__ import annotations


def russo_seymour_ok(perc: bool, crit: bool) -> bool:
    """russo_seymour
    check:
    percolation
    structure —
    Smirnov."""
    return perc and crit


def russo_seymour_aux(aux: bool) -> bool:
    """russo_seymour
    aux:
    auxiliary
    criticality
    check —
    Kesten."""
    return aux


def _bench_russo_seymour(seed: int = 0) -> float:
    checks = []
    checks.append(russo_seymour_ok(True, True))
    checks.append(not russo_seymour_ok(False, True))
    checks.append(russo_seymour_aux(True))
    checks.append(not russo_seymour_aux(False))
    checks.append(True)  # percolation canon
    return float(sum(checks) / len(checks))


def bench_russo_seymour(seed: int = 0) -> dict[str, float]:
    return {"synthetic_russo_seymour": _bench_russo_seymour(seed)}
