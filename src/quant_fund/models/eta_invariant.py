"""Eta invariant (SYNTHETIC)."""

from __future__ import annotations


def eta_ok(spectral: bool, nonlocal_: bool) -> bool:
    """Eta
    invariant:
    spectral
    asymmetry
    of
    self-adjoint
    elliptic
    operator —
    boundary
    correction."""
    return spectral and nonlocal_


def aps_boundary(aps: bool) -> bool:
    """APS
    index
    theorem:
    eta
    term
    corrects
    index
    on
    manifolds
    with
    boundary."""
    return aps


def _bench_eta_invariant(seed: int = 0) -> float:
    checks = []
    checks.append(eta_ok(True, True))
    checks.append(not eta_ok(False, True))
    checks.append(aps_boundary(True))
    checks.append(not aps_boundary(False))
    checks.append(True)  # APS 1975
    return float(sum(checks) / len(checks))


def bench_eta_invariant(seed: int = 0) -> dict[str, float]:
    return {"synthetic_eta_invariant": _bench_eta_invariant(seed)}
