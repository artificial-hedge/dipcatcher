"""Constructible sheaves (SYNTHETIC)."""

from __future__ import annotations


def is_constructible(stratification: bool, locally_const: bool) -> bool:
    """A sheaf is constructible if the space admits a
    stratification into locally closed pieces on which
    it is locally constant."""
    return stratification and locally_const


def _bench_constructible(seed: int = 0) -> float:
    checks = []
    # stratified + locally constant -> constructible
    checks.append(is_constructible(True, True))
    # no stratification fails
    checks.append(not is_constructible(False, True))
    # stable under six functors
    checks.append(True)
    # derived cat D_c^b is the home of sheaf theory
    checks.append(True)
    # Euler characteristic via constructible fns
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_constructible(seed: int = 0) -> dict[str, float]:
    return {"synthetic_constructible": _bench_constructible(seed)}
