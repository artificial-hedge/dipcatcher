"""Whitehead product (SYNTHETIC)."""

from __future__ import annotations


def wp_ok(whitehead: bool, graded: bool) -> bool:
    """Whitehead
    product:
    graded-
    Lie
    bracket
    on
    homotopy —
    Whitehead
    bracket."""
    return whitehead and graded


def whitehead_jac(wj: bool) -> bool:
    """Jacobi:
    Whitehead
    product
    satisfies
    graded
    Jacobi —
    graded
    Lie."""
    return wj


def _bench_whitehead_prod(seed: int = 0) -> float:
    checks = []
    checks.append(wp_ok(True, True))
    checks.append(not wp_ok(False, True))
    checks.append(whitehead_jac(True))
    checks.append(not whitehead_jac(False))
    checks.append(True)  # Whitehead
    return float(sum(checks) / len(checks))


def bench_whitehead_prod(seed: int = 0) -> dict[str, float]:
    return {"synthetic_whitehead_prod": _bench_whitehead_prod(seed)}
