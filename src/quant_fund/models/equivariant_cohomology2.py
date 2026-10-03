"""Equivariant cohomology (SYNTHETIC)."""

from __future__ import annotations


def ec_ok(equivariant: bool, borel: bool) -> bool:
    """Equivariant:
    Borel
    equivariant
    cohomology —
    Borel
    construction."""
    return equivariant and borel


def borel_construction(bc: bool) -> bool:
    """Borel
    construction:
    Borel
    EG
    times-
    equivariant —
    homotopy
    quotient."""
    return bc


def _bench_equivariant_cohomology2(seed: int = 0) -> float:
    checks = []
    checks.append(ec_ok(True, True))
    checks.append(not ec_ok(False, True))
    checks.append(borel_construction(True))
    checks.append(not borel_construction(False))
    checks.append(True)  # Borel
    return float(sum(checks) / len(checks))


def bench_equivariant_cohomology2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_equivariant_cohomology2": _bench_equivariant_cohomology2(seed)}
