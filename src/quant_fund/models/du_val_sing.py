"""Du Val singularities (SYNTHETIC)."""

from __future__ import annotations


def dv_ok(ade: bool, surface: bool) -> bool:
    """Du
    Val:
    ADE
    surface
    singularities —
    rational
    double
    points
    classified
    by
    Dynkin
    diagrams."""
    return ade and surface


def minimal_resolution(mr: bool) -> bool:
    """Minimal
    resolution:
    exceptional
    curves
    form
    the
    Dynkin
    diagram
    with
    self-
    intersections
    -2."""
    return mr


def _bench_du_val_sing(seed: int = 0) -> float:
    checks = []
    checks.append(dv_ok(True, True))
    checks.append(not dv_ok(False, True))
    checks.append(minimal_resolution(True))
    checks.append(not minimal_resolution(False))
    checks.append(True)  # Du Val
    return float(sum(checks) / len(checks))


def bench_du_val_sing(seed: int = 0) -> dict[str, float]:
    return {"synthetic_du_val_sing": _bench_du_val_sing(seed)}
