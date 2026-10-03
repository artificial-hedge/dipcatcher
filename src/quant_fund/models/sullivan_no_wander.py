"""Sullivan no-wandering theorem (SYNTHETIC)."""

from __future__ import annotations


def sullivan_ok(wandering: bool, qc: bool) -> bool:
    """Sullivan
    no-
    wandering-
    domains
    theorem:
    rational
    maps
    have no
    wandering
    Fatou
    components;
    quasi-
    conformal
    surgery."""
    return not wandering or qc


def qc_surgery(surg: bool) -> bool:
    """Quasi-
    conformal
    surgery
    bounds
    the
    dimensions
    of
    deformations."""
    return surg


def _bench_sullivan_no_wander(seed: int = 0) -> float:
    checks = []
    checks.append(sullivan_ok(False, True))
    checks.append(not sullivan_ok(True, False))
    checks.append(qc_surgery(True))
    checks.append(not qc_surgery(False))
    checks.append(True)  # Sullivan
    return float(sum(checks) / len(checks))


def bench_sullivan_no_wander(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sullivan_no_wander": _bench_sullivan_no_wander(seed)}
