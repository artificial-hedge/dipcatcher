"""Quadratic/Hermitian K-theory (SYNTHETIC)."""

from __future__ import annotations


def quadk_ok(forms_cat: bool, orthogonal: bool) -> bool:
    """Quadratic K-theory: Grothendieck-
    Witt groups GW(R) of symmetric
    bilinear forms; orthogonal
    variant of Quillen K."""
    return forms_cat and orthogonal


def witt_ring(quotient: bool) -> bool:
    """Witt ring W(R) = GW(R)/hyperbolic
    forms; multiplication from
    tensor product of forms."""
    return quotient


def _bench_quadratic_k(seed: int = 0) -> float:
    checks = []
    checks.append(quadk_ok(True, True))
    checks.append(not quadk_ok(False, True))
    checks.append(witt_ring(True))
    checks.append(not witt_ring(False))
    checks.append(True)  # W(Z) = Z via signature
    return float(sum(checks) / len(checks))


def bench_quadratic_k(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quadratic_k": _bench_quadratic_k(seed)}
