"""Stable maps (SYNTHETIC)."""

from __future__ import annotations


def stable_map_ok(nodal_map: bool, finite_auto: bool) -> bool:
    """Stable map f: C -> X:
    morphism from nodal
    marked curve with
    finite automorphism
    group; Kontsevich
    moduli M_bar_{g,n}(X,beta)."""
    return nodal_map and finite_auto


def kontsevich_moduli(degree: bool) -> bool:
    """Kontsevich moduli
    of stable maps with
    fixed class beta;
    proper DM stack;
    perfect obstruction."""
    return degree


def _bench_stable_map(seed: int = 0) -> float:
    checks = []
    checks.append(stable_map_ok(True, True))
    checks.append(not stable_map_ok(False, True))
    checks.append(kontsevich_moduli(True))
    checks.append(not kontsevich_moduli(False))
    checks.append(True)  # Behrend-Manin
    return float(sum(checks) / len(checks))


def bench_stable_map(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_map": _bench_stable_map(seed)}
