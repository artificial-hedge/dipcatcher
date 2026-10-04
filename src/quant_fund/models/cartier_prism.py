"""Cartier prism (SYNTHETIC)."""

from __future__ import annotations


def cp_ok(cartier: bool, prism: bool) -> bool:
    """Cartier
    prism:
    Cartier
    prism —
    de
    Rham."""
    return cartier and prism


def cartier_isom(ci: bool) -> bool:
    """Cartier
    isom:
    Cartier
    isomorphism —
    conjugate."""
    return ci


def _bench_cartier_prism(seed: int = 0) -> float:
    checks = []
    checks.append(cp_ok(True, True))
    checks.append(not cp_ok(False, True))
    checks.append(cartier_isom(True))
    checks.append(not cartier_isom(False))
    checks.append(True)  # Cartier
    return float(sum(checks) / len(checks))


def bench_cartier_prism(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cartier_prism": _bench_cartier_prism(seed)}
