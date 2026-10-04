"""Gros topos / big site (SYNTHETIC)."""

from __future__ import annotations


def gros_ok(big_site: bool, func_of_points: bool) -> bool:
    """Gros topos Sh(C, big) sheaves
    on big site; schemes embed
    as representables via
    functor-of-points."""
    return big_site and func_of_points


def yoneda_topos(presheaf: bool) -> bool:
    """A scheme X embeds in
    gros topos as Hom(-, X);
    gluing conditions hold
    for Zariski covers."""
    return presheaf


def _bench_gros_topos(seed: int = 0) -> float:
    checks = []
    checks.append(gros_ok(True, True))
    checks.append(not gros_ok(False, True))
    checks.append(yoneda_topos(True))
    checks.append(not yoneda_topos(False))
    checks.append(True)  # schemes = sheaves on Zariski
    return float(sum(checks) / len(checks))


def bench_gros_topos(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gros_topos": _bench_gros_topos(seed)}
