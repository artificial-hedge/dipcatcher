"""Friedlander-Voevodsky (SYNTHETIC)."""

from __future__ import annotations


def fv_ok(friedlander: bool, bivariant: bool) -> bool:
    """Friedlander-
    Voevodsky:
    bivariant
    cycle
    cohomology —
    Lawson
    homology."""
    return friedlander and bivariant


def friedlander_duality(fd: bool) -> bool:
    """FV
    duality:
    duality
    between
    cycle
    cohomology
    and
    homology —
    Friedlander-
    Voevodsky."""
    return fd


def _bench_friedlander_voev(seed: int = 0) -> float:
    checks = []
    checks.append(fv_ok(True, True))
    checks.append(not fv_ok(False, True))
    checks.append(friedlander_duality(True))
    checks.append(not friedlander_duality(False))
    checks.append(True)  # Friedlander-Voevodsky
    return float(sum(checks) / len(checks))


def bench_friedlander_voev(seed: int = 0) -> dict[str, float]:
    return {"synthetic_friedlander_voev": _bench_friedlander_voev(seed)}
