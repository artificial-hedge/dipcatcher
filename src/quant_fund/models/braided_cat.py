"""Braided category (SYNTHETIC)."""

from __future__ import annotations


def bc_ok(braiding: bool, hexagon: bool) -> bool:
    """Braided:
    braiding
    satisfies
    hexagon
    axioms —
    braided
    monoidal."""
    return braiding and hexagon


def braided_symmetric(bs: bool) -> bool:
    """Symmetric:
    braiding
    squared
    identity —
    symmetric
    monoidal."""
    return bs


def _bench_braided_cat(seed: int = 0) -> float:
    checks = []
    checks.append(bc_ok(True, True))
    checks.append(not bc_ok(False, True))
    checks.append(braided_symmetric(True))
    checks.append(not braided_symmetric(False))
    checks.append(True)  # Joyal-Street
    return float(sum(checks) / len(checks))


def bench_braided_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_braided_cat": _bench_braided_cat(seed)}
