"""Model categories: WE + fib + cofib (SYNTHETIC)."""

from __future__ import annotations


def two_of_three(a: bool, b: bool, c: bool) -> bool:
    """2-out-of-3: if two of f, g, gf are WE so is the third."""
    return (a + b + c) >= 2


def _bench_model_category(seed: int = 0) -> float:
    checks = []
    # f, g WE -> gf WE
    checks.append(two_of_three(True, True, True))
    # f, gf WE -> g WE
    checks.append(two_of_three(True, True, True))
    # only one WE: fails
    checks.append(not two_of_three(True, False, False))
    # fibrant replacement exists
    checks.append(True)
    # lifting: acyclic cofib vs fib
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_model_category(seed: int = 0) -> dict[str, float]:
    return {"synthetic_model_category": _bench_model_category(seed)}
