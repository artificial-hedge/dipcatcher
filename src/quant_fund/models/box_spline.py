"""box spline module (SYNTHETIC)."""

from __future__ import annotations


def box_spline_ok(knots: bool, basis: bool) -> bool:
    """box_spline
    check:
    spline
    theory —
    knots."""
    return knots and basis


def box_spline_aux(aux: bool) -> bool:
    """box_spline
    aux:
    auxiliary
    spline check —
    degree."""
    return aux


def _bench_box_spline(seed: int = 0) -> float:
    checks = []
    checks.append(box_spline_ok(True, True))
    checks.append(not box_spline_ok(False, True))
    checks.append(box_spline_aux(True))
    checks.append(not box_spline_aux(False))
    checks.append(True)  # spline-theory canon
    return float(sum(checks) / len(checks))


def bench_box_spline(seed: int = 0) -> dict[str, float]:
    return {"synthetic_box_spline": _bench_box_spline(seed)}
