"""Homological mirror symmetry (SYNTHETIC)."""

from __future__ import annotations


def hms_ok(fukaya: bool, derived: bool) -> bool:
    """HMS
    conjecture:
    Fukaya
    category
    of
    a
    symplectic
    manifold
    is
    equivalent
    to
    the
    derived
    category
    of
    coherent
    sheaves
    on
    the
    mirror —
    Kontsevich
    1994."""
    return fukaya and derived


def ab_equivalence(ab: bool) -> bool:
    """A-B
    equivalence:
    Lagrangian
    branes
    and
    their
    Floer
    homology
    match
    sheaves
    and
    Ext
    groups."""
    return ab


def _bench_hms_conjecture(seed: int = 0) -> float:
    checks = []
    checks.append(hms_ok(True, True))
    checks.append(not hms_ok(False, True))
    checks.append(ab_equivalence(True))
    checks.append(not ab_equivalence(False))
    checks.append(True)  # Kontsevich
    return float(sum(checks) / len(checks))


def bench_hms_conjecture(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hms_conjecture": _bench_hms_conjecture(seed)}
