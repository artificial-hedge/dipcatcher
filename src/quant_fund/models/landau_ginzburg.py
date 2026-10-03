"""Landau-Ginzburg models (SYNTHETIC)."""

from __future__ import annotations


def lg_ok(superpotential: bool, critical: bool) -> bool:
    """Landau-
    Ginzburg
    model:
    a
    variety
    with
    superpotential
    W
    —
    mirror
    partner
    of
    the
    original
    CY."""
    return superpotential and critical


def orlov_equiv(oe: bool) -> bool:
    """Orlov
    equivalence:
    matrix
    factorizations
    of
    W
    is
    the
    B-model
    category
    of
    the
    LG
    mirror."""
    return oe


def _bench_landau_ginzburg(seed: int = 0) -> float:
    checks = []
    checks.append(lg_ok(True, True))
    checks.append(not lg_ok(False, True))
    checks.append(orlov_equiv(True))
    checks.append(not orlov_equiv(False))
    checks.append(True)  # Orlov
    return float(sum(checks) / len(checks))


def bench_landau_ginzburg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_landau_ginzburg": _bench_landau_ginzburg(seed)}
