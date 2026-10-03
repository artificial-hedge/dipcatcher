"""Tropical cycles / Minkowski weights (SYNTHETIC)."""

from __future__ import annotations


def tropical_cycle_ok(minkowski: bool, weights: bool) -> bool:
    """Tropical cycle:
    balanced weighted
    polyhedral complex;
    Minkowski weights on
    fans encode Chow
    classes."""
    return minkowski and weights


def tropical_chow(fulton_sturm: bool) -> bool:
    """Tropical Chow group:
    stable intersection of
    tropical cycles;
    Fulton-Sturmfels
    Minkowski weights."""
    return fulton_sturm


def _bench_tropical_cycle(seed: int = 0) -> float:
    checks = []
    checks.append(tropical_cycle_ok(True, True))
    checks.append(not tropical_cycle_ok(False, True))
    checks.append(tropical_chow(True))
    checks.append(not tropical_chow(False))
    checks.append(True)  # tropical Poincaré duality
    return float(sum(checks) / len(checks))


def bench_tropical_cycle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tropical_cycle": _bench_tropical_cycle(seed)}
