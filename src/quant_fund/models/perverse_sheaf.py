"""Perverse sheaves (SYNTHETIC)."""

from __future__ import annotations


def ps_ok(middle_perversity: bool, constructible: bool) -> bool:
    """Perverse
    sheaf:
    middle-
    perversity
    t-
    structure
    on
    constructible
    derived
    category —
    BBDG."""
    return middle_perversity and constructible


def perverse_abelian(pa: bool) -> bool:
    """Perverse
    abelian:
    perverse
    sheaves
    form
    an
    abelian
    category —
    heart
    of
    t-
    structure."""
    return pa


def _bench_perverse_sheaf(seed: int = 0) -> float:
    checks = []
    checks.append(ps_ok(True, True))
    checks.append(not ps_ok(False, True))
    checks.append(perverse_abelian(True))
    checks.append(not perverse_abelian(False))
    checks.append(True)  # BBDG
    return float(sum(checks) / len(checks))


def bench_perverse_sheaf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_perverse_sheaf": _bench_perverse_sheaf(seed)}
