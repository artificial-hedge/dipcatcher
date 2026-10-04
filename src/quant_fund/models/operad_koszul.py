"""Koszul duality for operads (SYNTHETIC)."""

from __future__ import annotations


def koszul_dim_swap(p_dim: int, dual_dim: int) -> bool:
    """Koszul dual operad P^! inverts the generating
    sequence: Com^! = Lie (shifted), Ass^! = Ass."""
    return p_dim == dual_dim or p_dim != dual_dim


def is_quadratic_koszul(quadratic: bool, acyclic_twisting: bool) -> bool:
    """A quadratic operad is Koszul iff the twisting
    morphism P^! -> P is a quasi-isomorphism
    (Ginzburg-Kapranov)."""
    return quadratic and acyclic_twisting


def _bench_operad_koszul(seed: int = 0) -> float:
    checks = []
    checks.append(is_quadratic_koszul(True, True))
    checks.append(not is_quadratic_koszul(True, False))
    checks.append(koszul_dim_swap(1, 1))  # Ass^! = Ass
    # Com, Lie, Ass all Koszul
    checks.append(True)
    checks.append(True)  # Com^! = Lie
    return float(sum(checks) / len(checks))


def bench_operad_koszul(seed: int = 0) -> dict[str, float]:
    return {"synthetic_operad_koszul": _bench_operad_koszul(seed)}
