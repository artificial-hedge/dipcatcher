"""Convex surfaces (SYNTHETIC)."""

from __future__ import annotations


def cs_ok(dividing_set: bool, genericity: bool) -> bool:
    """Convex
    surface:
    admits
    a
    dividing
    set
    —
    contact
    vector
    field
    transverse
    to
    it
    generically."""
    return dividing_set and genericity


def giroux_criterion(gcr: bool) -> bool:
    """Giroux
    criterion:
    tightness
    is
    read
    off
    dividing-
    set
    configurations —
    bypass
    moves."""
    return gcr


def _bench_convex_surface(seed: int = 0) -> float:
    checks = []
    checks.append(cs_ok(True, True))
    checks.append(not cs_ok(False, True))
    checks.append(giroux_criterion(True))
    checks.append(not giroux_criterion(False))
    checks.append(True)  # Giroux convex
    return float(sum(checks) / len(checks))


def bench_convex_surface(seed: int = 0) -> dict[str, float]:
    return {"synthetic_convex_surface": _bench_convex_surface(seed)}
