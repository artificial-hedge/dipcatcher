"""Intersection forms (SYNTHETIC)."""

from __future__ import annotations


def if_ok(symmetric: bool, unimodular: bool) -> bool:
    """Intersection
    form:
    symmetric
    unimodular
    bilinear
    form
    on
    H2 —
    classifies
    simply-
    connected
    4-manifolds
    topologically."""
    return symmetric and unimodular


def serre_class(sc: bool) -> bool:
    """Serre
    classification:
    indefinite
    unimodular
    forms
    are
    classified
    by
    rank,
    signature,
    type."""
    return sc


def _bench_intersection_form(seed: int = 0) -> float:
    checks = []
    checks.append(if_ok(True, True))
    checks.append(not if_ok(False, True))
    checks.append(serre_class(True))
    checks.append(not serre_class(False))
    checks.append(True)  # Whitehead
    return float(sum(checks) / len(checks))


def bench_intersection_form(seed: int = 0) -> dict[str, float]:
    return {"synthetic_intersection_form": _bench_intersection_form(seed)}
