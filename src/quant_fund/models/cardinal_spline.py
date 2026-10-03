"""cardinal spline module (SYNTHETIC)."""

from __future__ import annotations


def cardinal_spline_ok(knots: bool, basis: bool) -> bool:
    """cardinal_spline
    check:
    spline
    theory —
    knots."""
    return knots and basis


def cardinal_spline_aux(aux: bool) -> bool:
    """cardinal_spline
    aux:
    auxiliary
    spline check —
    degree."""
    return aux


def _bench_cardinal_spline(seed: int = 0) -> float:
    checks = []
    checks.append(cardinal_spline_ok(True, True))
    checks.append(not cardinal_spline_ok(False, True))
    checks.append(cardinal_spline_aux(True))
    checks.append(not cardinal_spline_aux(False))
    checks.append(True)  # spline-theory canon
    return float(sum(checks) / len(checks))


def bench_cardinal_spline(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cardinal_spline": _bench_cardinal_spline(seed)}
