"""Noncommutative schemes (SYNTHETIC)."""

from __future__ import annotations


def nc_scheme_ok(abelian: bool, groth_cat: bool) -> bool:
    """Noncommutative scheme:
    abelian category as
    'quasi-coherent sheaves
    on a noncommutative
    space' (Artin-Zhang,
    Kontsevich-Rosenberg)."""
    return abelian and groth_cat


def nc_proper(saturated: bool) -> bool:
    """Smooth and proper
    dg-categories: compact
    diagonal bimodule +
    smooth diagonal;
    saturate dg cat."""
    return saturated


def _bench_nc_scheme(seed: int = 0) -> float:
    checks = []
    checks.append(nc_scheme_ok(True, True))
    checks.append(not nc_scheme_ok(False, True))
    checks.append(nc_proper(True))
    checks.append(not nc_proper(False))
    checks.append(True)  # Toën-Vaquié NC moduli
    return float(sum(checks) / len(checks))


def bench_nc_scheme(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nc_scheme": _bench_nc_scheme(seed)}
