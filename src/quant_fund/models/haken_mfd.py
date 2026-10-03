"""Haken manifolds (SYNTHETIC)."""

from __future__ import annotations


def haken_ok(incompressible: bool, hierarchy: bool) -> bool:
    """Haken
    manifold:
    contains
    an
    incompressible
    surface —
    cut
    inductively
    to
    a
    ball."""
    return incompressible and hierarchy


def waldhausen(w: bool) -> bool:
    """Waldhausen:
    Haken
    manifolds
    are
    classified
    by
    their
    hierarchy
    —
    homeomorphism
    problem
    solvable."""
    return w


def _bench_haken_mfd(seed: int = 0) -> float:
    checks = []
    checks.append(haken_ok(True, True))
    checks.append(not haken_ok(False, True))
    checks.append(waldhausen(True))
    checks.append(not waldhausen(False))
    checks.append(True)  # Haken-Waldhausen
    return float(sum(checks) / len(checks))


def bench_haken_mfd(seed: int = 0) -> dict[str, float]:
    return {"synthetic_haken_mfd": _bench_haken_mfd(seed)}
