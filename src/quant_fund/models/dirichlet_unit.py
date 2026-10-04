"""Dirichlet unit theorem: rank r1 + r2 - 1 (SYNTHETIC)."""

from __future__ import annotations


def unit_rank(r1: int, r2: int) -> int:
    """O_K^* ~ mu(K) x Z^{r1 + r2 - 1}."""
    return r1 + r2 - 1


def _bench_dirichlet_unit(seed: int = 0) -> float:
    checks = []
    # Q: rank 0 (only +-1)
    checks.append(unit_rank(1, 0) == 0)
    # imaginary quadratic: rank 0
    checks.append(unit_rank(0, 1) == 0)
    # real quadratic Q(sqrt(2)): rank 1 (fundamental unit)
    checks.append(unit_rank(2, 0) == 1)
    # totally real cubic: rank 2
    checks.append(unit_rank(3, 0) == 2)
    # CM field rank: r2 - 1
    checks.append(unit_rank(0, 2) == 1)
    return float(sum(checks) / len(checks))


def bench_dirichlet_unit(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dirichlet_unit": _bench_dirichlet_unit(seed)}
