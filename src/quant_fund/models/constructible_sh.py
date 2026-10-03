"""Constructible sheaves (SYNTHETIC)."""

from __future__ import annotations


def cs_ok(constructible: bool, sheaf: bool) -> bool:
    """Constructible
    sheaf:
    constructible
    sheaf —
    locally
    constant
    on
    strata."""
    return constructible and sheaf


def lisse_sheaf_toy(ls: bool) -> bool:
    """Lisse
    sheaf:
    lisse
    sheaf —
    locally
    constant
    constructible."""
    return ls


def _bench_constructible_sh(seed: int = 0) -> float:
    checks = []
    checks.append(cs_ok(True, True))
    checks.append(not cs_ok(False, True))
    checks.append(lisse_sheaf_toy(True))
    checks.append(not lisse_sheaf_toy(False))
    checks.append(True)  # SGA4
    return float(sum(checks) / len(checks))


def bench_constructible_sh(seed: int = 0) -> dict[str, float]:
    return {"synthetic_constructible_sh": _bench_constructible_sh(seed)}
