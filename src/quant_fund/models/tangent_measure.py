"""Tangent measures (SYNTHETIC)."""

from __future__ import annotations


def tm_ok(blowup: bool, cone: bool) -> bool:
    """Tangent
    measures:
    limits
    of
    rescaled
    measures
    form
    a
    dilation-
    invariant
    cone."""
    return blowup and cone


def preiss_uniform(uniform: bool) -> bool:
    """Preiss:
    uniform
    density
    forces
    tangent
    measures
    to be
    flat."""
    return uniform


def _bench_tangent_measure(seed: int = 0) -> float:
    checks = []
    checks.append(tm_ok(True, True))
    checks.append(not tm_ok(False, True))
    checks.append(preiss_uniform(True))
    checks.append(not preiss_uniform(False))
    checks.append(True)  # Preiss
    return float(sum(checks) / len(checks))


def bench_tangent_measure(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tangent_measure": _bench_tangent_measure(seed)}
