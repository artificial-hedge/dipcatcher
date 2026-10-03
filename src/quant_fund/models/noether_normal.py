"""Noether normalization bookkeeping (SYNTHETIC)."""

from __future__ import annotations


def normalization_vars(rel_weights: list[int], coeffs: list[int]) -> int:
    """Number of algebraically independent elements in a Noether
    normalization of a graded ring with a weighted relation: for a
    hypersurface in A^n it's n - 1."""
    n = len(coeffs)
    return n - 1 if any(rel_weights) else n


def integral_degree(f_deg: int, n_vars: int) -> int:
    """Finite extension degree of k[x_1..x_{n-1}] -> R for a monic
    relation of degree f_deg: the fiber size model."""
    return f_deg


def _bench_noether_normal(seed: int = 0) -> float:
    checks = []
    # k[x,y]/(y^2 - x^3): normalization k[x] -> one var, degree 2
    checks.append(normalization_vars([1], [1, 1]) == 1)
    checks.append(integral_degree(2, 2) == 2)
    # smooth quadric surface in A^3: 2 free vars
    checks.append(normalization_vars([1], [1, 1, 1]) == 2)
    # no relation: all n vars independent
    checks.append(normalization_vars([0], [1, 1, 1]) == 3)
    # curve y = x^2: still one free var, degree 2 fiber
    checks.append(normalization_vars([1], [1, 1]) == 1)
    checks.append(integral_degree(2, 1) == 2)
    return float(sum(checks) / len(checks))


def bench_noether_normal(seed: int = 0) -> dict[str, float]:
    return {"synthetic_noether_normal": _bench_noether_normal(seed)}
