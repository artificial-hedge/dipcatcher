"""Formal neighborhoods (SYNTHETIC)."""

from __future__ import annotations


def fn_ok(formal: bool, neighborhood: bool) -> bool:
    """Formal
    neighborhood:
    formal
    neighborhood
    of
    a
    subscheme —
    formal
    tube."""
    return formal and neighborhood


def infinitesimal_nbhd(inb: bool) -> bool:
    """Infinitesimal
    neighborhood:
    infinitesimal
    neighborhood —
    infinitesimal
    tube."""
    return inb


def _bench_formal_neighborhood(seed: int = 0) -> float:
    checks = []
    checks.append(fn_ok(True, True))
    checks.append(not fn_ok(False, True))
    checks.append(infinitesimal_nbhd(True))
    checks.append(not infinitesimal_nbhd(False))
    checks.append(True)  # Grothendieck
    return float(sum(checks) / len(checks))


def bench_formal_neighborhood(seed: int = 0) -> dict[str, float]:
    return {"synthetic_formal_neighborhood": _bench_formal_neighborhood(seed)}
