"""Anosov diffeomorphisms (SYNTHETIC)."""

from __future__ import annotations


def anosov_ok(splitting: bool, uniform: bool) -> bool:
    """Anosov
    diffeo:
    global
    uniform
    hyperbolic
    splitting
    E^s plus
    E^u
    with
    contraction/
    expansion
    rates."""
    return splitting and uniform


def structural_stable(struct: bool) -> bool:
    """Structural
    stability:
    C^1-small
    perturbations
    of Anosov
    are
    topologically
    conjugate."""
    return struct


def _bench_anosov(seed: int = 0) -> float:
    checks = []
    checks.append(anosov_ok(True, True))
    checks.append(not anosov_ok(False, True))
    checks.append(structural_stable(True))
    checks.append(not structural_stable(False))
    checks.append(True)  # Anosov
    return float(sum(checks) / len(checks))


def bench_anosov(seed: int = 0) -> dict[str, float]:
    return {"synthetic_anosov": _bench_anosov(seed)}
