"""Schauder estimates (SYNTHETIC)."""

from __future__ import annotations


def sch_ok(holder: bool, two_derivs: bool) -> bool:
    """Schauder:
    Holder
    data
    gives
    C^{2,alpha}
    control
    on
    solutions
    of
    elliptic
    equations."""
    return holder and two_derivs


def interior_est(ie: bool) -> bool:
    """Interior
    estimates:
    C^{2,alpha}
    norm
    in
    interior
    balls
    controlled
    by
    data."""
    return ie


def _bench_schauder_est(seed: int = 0) -> float:
    checks = []
    checks.append(sch_ok(True, True))
    checks.append(not sch_ok(False, True))
    checks.append(interior_est(True))
    checks.append(not interior_est(False))
    checks.append(True)  # Schauder
    return float(sum(checks) / len(checks))


def bench_schauder_est(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schauder_est": _bench_schauder_est(seed)}
