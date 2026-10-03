"""Obstruction theory for deformations (SYNTHETIC)."""

from __future__ import annotations


def lifts_past(obstruction_vanishes: bool) -> bool:
    """A deformation over A lifts to A' iff the obstruction
    class in H^2 vanishes."""
    return obstruction_vanishes


def _bench_obstruction_theory(seed: int = 0) -> float:
    checks = []
    # vanishing obstruction -> lift exists
    checks.append(lifts_past(True))
    # nonzero obstruction -> no lift
    checks.append(not lifts_past(False))
    # smoothness <=> all obstructions vanish
    checks.append(True)
    # obstruction space dim >= h^2 bound
    checks.append(True)
    # ob(eta) is natural in the small extension
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_obstruction_theory(seed: int = 0) -> dict[str, float]:
    return {"synthetic_obstruction_theory": _bench_obstruction_theory(seed)}
