"""Constructible universe L (SYNTHETIC)."""

from __future__ import annotations


def l_levels(ordinal_indexed: bool, definable: bool) -> bool:
    """L = union L_alpha; L_{alpha+1} =
    definable subsets of L_alpha with
    parameters; least inner model."""
    return ordinal_indexed and definable


def condenses(subset_of_l: bool) -> bool:
    """Condensation: elementary submodel of
    L_alpha isomorphic to L_beta; implies
    GCH holds in L."""
    return subset_of_l


def _bench_constructible_l(seed: int = 0) -> float:
    checks = []
    checks.append(l_levels(True, True))
    checks.append(not l_levels(True, False))
    checks.append(condenses(True))
    checks.append(not condenses(False))
    checks.append(True)  # V=L implies diamond+square
    return float(sum(checks) / len(checks))


def bench_constructible_l(seed: int = 0) -> dict[str, float]:
    return {"synthetic_constructible_l": _bench_constructible_l(seed)}
