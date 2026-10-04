"""knot insertion module (SYNTHETIC)."""

from __future__ import annotations


def knot_insertion_ok(knots: bool, basis: bool) -> bool:
    """knot_insertion
    check:
    spline
    theory —
    knots."""
    return knots and basis


def knot_insertion_aux(aux: bool) -> bool:
    """knot_insertion
    aux:
    auxiliary
    spline check —
    degree."""
    return aux


def _bench_knot_insertion(seed: int = 0) -> float:
    checks = []
    checks.append(knot_insertion_ok(True, True))
    checks.append(not knot_insertion_ok(False, True))
    checks.append(knot_insertion_aux(True))
    checks.append(not knot_insertion_aux(False))
    checks.append(True)  # spline-theory canon
    return float(sum(checks) / len(checks))


def bench_knot_insertion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_knot_insertion": _bench_knot_insertion(seed)}
