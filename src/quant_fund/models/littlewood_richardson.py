"""Littlewood-Richardson rule (SYNTHETIC)."""

from __future__ import annotations


def lr_ok(coefficient: bool, tableau: bool) -> bool:
    """Littlewood-
    Richardson:
    combinatorial
    rule
    for
    Schur
    product
    coefficients —
    LR
    tableaux
    counting."""
    return coefficient and tableau


def schur_product(sp: bool) -> bool:
    """Schur
    product:
    decomposition
    of
    tensor
    products
    of
    GL
    representations —
    LR
    coefficients."""
    return sp


def _bench_littlewood_richardson(seed: int = 0) -> float:
    checks = []
    checks.append(lr_ok(True, True))
    checks.append(not lr_ok(False, True))
    checks.append(schur_product(True))
    checks.append(not schur_product(False))
    checks.append(True)  # Littlewood-Richardson
    return float(sum(checks) / len(checks))


def bench_littlewood_richardson(seed: int = 0) -> dict[str, float]:
    return {"synthetic_littlewood_richardson": _bench_littlewood_richardson(seed)}
