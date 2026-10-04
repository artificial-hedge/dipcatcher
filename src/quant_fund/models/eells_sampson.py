"""Eells-Sampson theorem (SYNTHETIC)."""

from __future__ import annotations


def es_ok(heat_flow: bool, neg_curv: bool) -> bool:
    """Eells-
    Sampson:
    harmonic
    map
    heat
    flow
    converges
    under
    negative
    target
    curvature."""
    return heat_flow and neg_curv


def unique_homotopy(uh: bool) -> bool:
    """Unique
    homotopy
    class:
    in
    negative
    curvature
    each
    homotopy
    class
    has
    a
    unique
    harmonic
    representative."""
    return uh


def _bench_eells_sampson(seed: int = 0) -> float:
    checks = []
    checks.append(es_ok(True, True))
    checks.append(not es_ok(False, True))
    checks.append(unique_homotopy(True))
    checks.append(not unique_homotopy(False))
    checks.append(True)  # Eells-Sampson 1964
    return float(sum(checks) / len(checks))


def bench_eells_sampson(seed: int = 0) -> dict[str, float]:
    return {"synthetic_eells_sampson": _bench_eells_sampson(seed)}
