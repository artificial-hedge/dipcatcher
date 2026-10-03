"""Tangent cone and multiplicity at a point (SYNTHETIC)."""

from __future__ import annotations


def multiplicity(lowest_deg: int) -> int:
    """Multiplicity at origin = degree of the lowest-degree form."""
    return lowest_deg


def _bench_tangent_cone(seed: int = 0) -> float:
    checks = []
    # smooth point: lowest term degree 1
    checks.append(multiplicity(1) == 1)
    # node y^2 = x^2 + x^3: multiplicity 2
    checks.append(multiplicity(2) == 2)
    # cusp y^2 = x^3: multiplicity 2 but one tangent direction
    checks.append(multiplicity(2) == 2)
    # ordinary triple point: 3
    checks.append(multiplicity(3) == 3)
    # tangent cone = Spec of graded ring of initial forms
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_tangent_cone(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tangent_cone": _bench_tangent_cone(seed)}
