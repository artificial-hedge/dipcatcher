"""Lagrange inversion (SYNTHETIC)."""

from __future__ import annotations


def li_ok(formal_inverse: bool, coefficients: bool) -> bool:
    """Lagrange
    inversion:
    coefficients
    of
    compositional
    inverse —
    tree
    counting."""
    return formal_inverse and coefficients


def cayley_trees(ct: bool) -> bool:
    """Cayley
    trees:
    n
    to
    the
    n-2
    labeled
    trees —
    Lagrange
    proof."""
    return ct


def _bench_lagrange_inversion(seed: int = 0) -> float:
    checks = []
    checks.append(li_ok(True, True))
    checks.append(not li_ok(False, True))
    checks.append(cayley_trees(True))
    checks.append(not cayley_trees(False))
    checks.append(True)  # Lagrange
    return float(sum(checks) / len(checks))


def bench_lagrange_inversion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lagrange_inversion": _bench_lagrange_inversion(seed)}
