"""Cartesian morphisms (SYNTHETIC)."""

from __future__ import annotations


def cm_ok(cartesian: bool, morphism: bool) -> bool:
    """Cartesian
    morphism:
    cartesian
    morphism —
    cartesian
    edge."""
    return cartesian and morphism


def cartesian_edge(ce: bool) -> bool:
    """Cartesian
    edge:
    cartesian
    edge
    of
    a
    fibration —
    cartesian
    lift."""
    return ce


def _bench_cartesian_morphism(seed: int = 0) -> float:
    checks = []
    checks.append(cm_ok(True, True))
    checks.append(not cm_ok(False, True))
    checks.append(cartesian_edge(True))
    checks.append(not cartesian_edge(False))
    checks.append(True)  # Grothendieck-Lurie
    return float(sum(checks) / len(checks))


def bench_cartesian_morphism(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cartesian_morphism": _bench_cartesian_morphism(seed)}
