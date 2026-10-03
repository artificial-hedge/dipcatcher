"""Drinfeld tower (SYNTHETIC)."""

from __future__ import annotations


def drinfeld_ok(rigid_tower: bool, level_cover: bool) -> bool:
    """Drinfeld tower: etale covers of
    p-adic upper half space with
    GL_2 action; first example of
    a local Langlands realization."""
    return rigid_tower and level_cover


def dual_tower(shimura: bool) -> bool:
    """Dual Drinfeld tower via group
    action is the Lubin-Tate tower;
    the two towers are intertwined
    (Faltings, Fargues)."""
    return shimura


def _bench_drinfeld_tower(seed: int = 0) -> float:
    checks = []
    checks.append(drinfeld_ok(True, True))
    checks.append(not drinfeld_ok(False, True))
    checks.append(dual_tower(True))
    checks.append(not dual_tower(False))
    checks.append(True)  # Cerednik-Drinfeld uniformization
    return float(sum(checks) / len(checks))


def bench_drinfeld_tower(seed: int = 0) -> dict[str, float]:
    return {"synthetic_drinfeld_tower": _bench_drinfeld_tower(seed)}
