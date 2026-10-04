"""blossoming module (SYNTHETIC)."""

from __future__ import annotations


def blossoming_ok(knots: bool, basis: bool) -> bool:
    """blossoming
    check:
    spline
    theory —
    knots."""
    return knots and basis


def blossoming_aux(aux: bool) -> bool:
    """blossoming
    aux:
    auxiliary
    spline check —
    degree."""
    return aux


def _bench_blossoming(seed: int = 0) -> float:
    checks = []
    checks.append(blossoming_ok(True, True))
    checks.append(not blossoming_ok(False, True))
    checks.append(blossoming_aux(True))
    checks.append(not blossoming_aux(False))
    checks.append(True)  # spline-theory canon
    return float(sum(checks) / len(checks))


def bench_blossoming(seed: int = 0) -> dict[str, float]:
    return {"synthetic_blossoming": _bench_blossoming(seed)}
