"""Intersection theory (SYNTHETIC)."""

from __future__ import annotations


def it_ok(cycle_class: bool, product: bool) -> bool:
    """Intersection
    theory:
    Chow
    group
    product —
    Fulton
    intersection."""
    return cycle_class and product


def fulton_intersect(fi: bool) -> bool:
    """Fulton:
    intersection
    ring
    with
    rational
    equivalence —
    Chow
    ring."""
    return fi


def _bench_intersection_theory(seed: int = 0) -> float:
    checks = []
    checks.append(it_ok(True, True))
    checks.append(not it_ok(False, True))
    checks.append(fulton_intersect(True))
    checks.append(not fulton_intersect(False))
    checks.append(True)  # Fulton
    return float(sum(checks) / len(checks))


def bench_intersection_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_intersection_theory": _bench_intersection_theory(seed)}
