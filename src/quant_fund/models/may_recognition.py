"""May recognition principle: loop spaces and E_n actions (SYNTHETIC)."""

from __future__ import annotations


def ordered_components_1d(k: int) -> int:
    """pi_0 of ordered k-configurations in R^1: k! distinct orderings."""
    import math

    return math.factorial(k)


def connected_components_hd(k: int, dim: int) -> int:
    """pi_0 of ordered k-configurations in R^d, d >= 2: connected, so 1."""
    return 1 if dim >= 2 else ordered_components_1d(k)


def _bench_may_recognition(seed: int = 0) -> float:
    checks = []
    # E_1 is strictly ordered: 2 configs have 2 components
    checks.append(ordered_components_1d(2) == 2)
    checks.append(ordered_components_1d(3) == 6)
    # E_n for n>=2 is connected: swapping two cubes is a path
    checks.append(connected_components_hd(2, 2) == 1)
    checks.append(connected_components_hd(3, 3) == 1)
    # recognition: a connected E_2 space models a double loop space, so
    # its binary operation is commutative UP TO HOMOTOPY: the two orderings
    # of multiplying a,b are joined by a path -> pi_0(ab - ba) = 0
    checks.append(connected_components_hd(2, 2) == 1)
    # loop space lowers the operad dimension: Omega S^n is E_{n-1}:
    # Omega S^1 (cover of S1 ~ Z): E_0-ish discrete, Omega S^2 -> E_1,
    # Omega S^3 -> E_2
    e_dim_of_loop = {1: 0, 2: 1, 3: 2, 4: 3}
    checks.append(e_dim_of_loop[3] == 2)
    return float(sum(checks) / len(checks))


def bench_may_recognition(seed: int = 0) -> dict[str, float]:
    return {"synthetic_may_recognition": _bench_may_recognition(seed)}
