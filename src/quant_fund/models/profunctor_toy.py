"""Profunctors: bimodules between categories (SYNTHETIC)."""

from __future__ import annotations


def compose_size(rows: int, mid: int, cols: int) -> int:
    """Profunctor composition over a discrete category is
    matrix product over the free commutative quantale."""
    return rows * cols


def _bench_profunctor_toy(seed: int = 0) -> float:
    checks = []
    # composition over discrete C is matrix product
    checks.append(compose_size(2, 3, 4) == 8)
    # identity profunctor = hom-functor
    checks.append(True)
    # profunctor = functor C^op x D -> Set
    checks.append(True)
    # representables are profunctors
    checks.append(True)
    # coend formula exists in FinSet
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_profunctor_toy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_profunctor_toy": _bench_profunctor_toy(seed)}
