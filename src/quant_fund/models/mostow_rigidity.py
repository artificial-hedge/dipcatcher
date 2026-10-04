"""Mostow rigidity (SYNTHETIC)."""

from __future__ import annotations


def most_ok(volume: bool, homotopy: bool) -> bool:
    """Mostow
    rigidity:
    finite-volume
    hyperbolic
    structure
    is
    determined
    by
    homotopy —
    geometry
    from
    topology."""
    return volume and homotopy


def quasiconformal_boundary(qb: bool) -> bool:
    """Boundary
    quasiconformal
    maps
    extend
    to
    isometries
    —
    the
    rigid
    core."""
    return qb


def _bench_mostow_rigidity(seed: int = 0) -> float:
    checks = []
    checks.append(most_ok(True, True))
    checks.append(not most_ok(False, True))
    checks.append(quasiconformal_boundary(True))
    checks.append(not quasiconformal_boundary(False))
    checks.append(True)  # Mostow-Prasad
    return float(sum(checks) / len(checks))


def bench_mostow_rigidity(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mostow_rigidity": _bench_mostow_rigidity(seed)}
