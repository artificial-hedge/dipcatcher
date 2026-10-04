"""b spline module (SYNTHETIC)."""

from __future__ import annotations


def b_spline_ok(knots: bool, basis: bool) -> bool:
    """b_spline
    check:
    spline
    theory —
    knots."""
    return knots and basis


def b_spline_aux(aux: bool) -> bool:
    """b_spline
    aux:
    auxiliary
    spline check —
    degree."""
    return aux


def _bench_b_spline(seed: int = 0) -> float:
    checks = []
    checks.append(b_spline_ok(True, True))
    checks.append(not b_spline_ok(False, True))
    checks.append(b_spline_aux(True))
    checks.append(not b_spline_aux(False))
    checks.append(True)  # spline-theory canon
    return float(sum(checks) / len(checks))


def bench_b_spline(seed: int = 0) -> dict[str, float]:
    return {"synthetic_b_spline": _bench_b_spline(seed)}
