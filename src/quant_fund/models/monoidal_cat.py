"""Monoidal structure of (FinSet, x, 1) (SYNTHETIC)."""

from __future__ import annotations

import itertools
from collections.abc import Iterable


def product_set(
    x: Iterable[object], y: Iterable[object]
) -> set[tuple[object, object]]:
    return set(itertools.product(x, y))


def assoc_bij(
    x: Iterable[object], y: Iterable[object], z: Iterable[object]
) -> dict[tuple[object, object], tuple[object, tuple[object, object]]]:
    """((x,y),z) -> (x,(y,z)) is a bijection."""
    left = set(itertools.product(set(itertools.product(x, y)), z))
    return {((a, b), c): (a, (b, c)) for ((a, b), c) in left}


def _bench_monoidal_cat(seed: int = 0) -> float:
    checks = []
    x, y, z = {0, 1}, {2}, {3, 4}
    # product cardinalities multiply
    checks.append(len(product_set(x, y)) == 2)
    checks.append(len(product_set(product_set(x, y), z)) == 4)
    # associator is a bijection
    a = assoc_bij(x, y, z)
    checks.append(len(a) == 4)
    checks.append(len(set(a.values())) == 4)
    # unitor: 1 x X ~ X
    unit = product_set({0}, x)
    checks.append(len(unit) == len(x))
    # symmetry: X x Y ~ Y x X
    checks.append(len(product_set(x, y)) == len(product_set(y, x)))
    return float(sum(checks) / len(checks))


def bench_monoidal_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_monoidal_cat": _bench_monoidal_cat(seed)}
