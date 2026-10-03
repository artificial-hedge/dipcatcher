"""Flat-map fiber dimension formula (SYNTHETIC)."""

from __future__ import annotations


def fiber_dim(dim_total: int, dim_base: int) -> int:
    """For a flat map X -> Y: dim generic fiber = dim X - dim Y."""
    return dim_total - dim_base


def add_vars(base_dim: int, n_new: int) -> int:
    """A[x_1..x_n] adds n to the dimension: dim A[x] = dim A + 1."""
    return base_dim + n_new


def _bench_dimension_fiber(seed: int = 0) -> float:
    checks = []
    # A^2 -> A^1 projection: generic fiber is a line, dim 1
    checks.append(fiber_dim(2, 1) == 1)
    # projection A^3 -> A^2: fiber dim 1
    checks.append(fiber_dim(3, 2) == 1)
    # adding a polynomial variable increments dimension
    checks.append(add_vars(2, 1) == 3)
    checks.append(add_vars(0, 2) == 2)
    # finite map: fiber dim 0
    checks.append(fiber_dim(1, 1) == 0)
    # product: dim(X x Y) = dim X + dim Y
    checks.append(add_vars(2, 1) - 2 == 1)
    return float(sum(checks) / len(checks))


def bench_dimension_fiber(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dimension_fiber": _bench_dimension_fiber(seed)}
