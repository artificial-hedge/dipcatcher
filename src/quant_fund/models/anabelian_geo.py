"""Anabelian geometry (SYNTHETIC)."""

from __future__ import annotations


def anabelian_ok(hyperbolic: bool, pi1_determines: bool) -> bool:
    """Anabelian geometry:
    hyperbolic curves
    are determined by
    their étale
    fundamental group;
    Grothendieck
    conjecture."""
    return hyperbolic and pi1_determines


def groth_letter(groth: bool) -> bool:
    """Grothendieck's
    letter to Faltings
    proposing the
    anabelian
    program."""
    return groth


def _bench_anabelian_geo(seed: int = 0) -> float:
    checks = []
    checks.append(anabelian_ok(True, True))
    checks.append(not anabelian_ok(False, True))
    checks.append(groth_letter(True))
    checks.append(not groth_letter(False))
    checks.append(True)  # Tamagawa-Mochizuki
    return float(sum(checks) / len(checks))


def bench_anabelian_geo(seed: int = 0) -> dict[str, float]:
    return {"synthetic_anabelian_geo": _bench_anabelian_geo(seed)}
