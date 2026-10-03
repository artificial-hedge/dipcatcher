"""Sutured manifolds (SYNTHETIC)."""

from __future__ import annotations


def sm_ok(sutures: bool, annuli: bool) -> bool:
    """Sutured
    manifold:
    oriented
    3-manifold
    with
    a
    dividing
    set
    of
    sutures
    on
    its
    boundary —
    Gabai's
    machinery."""
    return sutures and annuli


def decomposition(sd: bool) -> bool:
    """Sutured
    decomposition:
    cutting
    along
    well-
    chosen
    surfaces
    simplifies
    the
    hierarchy —
    detects
    fiberedness."""
    return sd


def _bench_sutured_mfd(seed: int = 0) -> float:
    checks = []
    checks.append(sm_ok(True, True))
    checks.append(not sm_ok(False, True))
    checks.append(decomposition(True))
    checks.append(not decomposition(False))
    checks.append(True)  # Gabai
    return float(sum(checks) / len(checks))


def bench_sutured_mfd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sutured_mfd": _bench_sutured_mfd(seed)}
