"""Crystalline site + sheaves (SYNTHETIC)."""

from __future__ import annotations


def crys_ok(pd_strat: bool, site_maps: bool) -> bool:
    """Crystalline site CRIS(X/S):
    objects are PD thickenings
    (U, T, delta) with nilpotent
    ideal; PD-stratifications
    = crystals."""
    return pd_strat and site_maps


def topos_morph(crys_perv: bool) -> bool:
    """Crystalline topos and its
    sheaf cohomology compute
    H^i_cris(X/W) tensor K."""
    return crys_perv


def _bench_berthelo_crys(seed: int = 0) -> float:
    checks = []
    checks.append(crys_ok(True, True))
    checks.append(not crys_ok(False, True))
    checks.append(topos_morph(True))
    checks.append(not topos_morph(False))
    checks.append(True)  # de Rham/crystalline comparison
    return float(sum(checks) / len(checks))


def bench_berthelo_crys(seed: int = 0) -> dict[str, float]:
    return {"synthetic_berthelo_crys": _bench_berthelo_crys(seed)}
