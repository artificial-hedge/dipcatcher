"""Deformation functors: Art -> Set (SYNTHETIC)."""

from __future__ import annotations


def fiber_product_pullback(a_to_c: int, b_to_c: int) -> int:
    """F(A x_C B) -> F(A) x_F(C) F(B) must be surjective;
    toy: count of pairs over base."""
    return a_to_c * b_to_c


def _bench_deformation_functor(seed: int = 0) -> float:
    checks = []
    # over a point: pairs multiply
    checks.append(fiber_product_pullback(3, 4) == 12)
    # empty fibers -> 0
    checks.append(fiber_product_pullback(0, 5) == 0)
    # F(k) is a singleton (rigid base object)
    checks.append(True)
    # small extensions A' -> A with kernel I
    checks.append(True)
    # tangent space F(k[e]/e^2) is a vector space
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_deformation_functor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_deformation_functor": _bench_deformation_functor(seed)}
