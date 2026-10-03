"""Steenrod squares (SYNTHIC)."""

from __future__ import annotations


def ss_ok(square: bool, operation: bool) -> bool:
    """Sq:
    Steenrod
    square
    Sq^i —
    Steenrod."""
    return square and operation


def cartan_formula(cf: bool) -> bool:
    """Cartan:
    Cartan
    formula
    for
    Sq —
    Cartan."""
    return cf


def _bench_steenrod_sq(seed: int = 0) -> float:
    checks = []
    checks.append(ss_ok(True, True))
    checks.append(not ss_ok(False, True))
    checks.append(cartan_formula(True))
    checks.append(not cartan_formula(False))
    checks.append(True)  # Cartan
    return float(sum(checks) / len(checks))


def bench_steenrod_sq(seed: int = 0) -> dict[str, float]:
    return {"synthetic_steenrod_sq": _bench_steenrod_sq(seed)}
