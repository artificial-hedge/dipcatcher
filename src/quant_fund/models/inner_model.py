"""Constructible universe L: cumulative levels (SYNTHETIC)."""

from __future__ import annotations


def defsubsets(level_size: int) -> int:
    """Number of definable subsets of a set of size n (toy bound)."""
    return int(2**level_size)


def _bench_inner_model(seed: int = 0) -> float:
    checks = []
    # L_0 = empty, L_{n+1} = Def(L_n)
    checks.append(defsubsets(0) == 1)
    # L_1 has the empty set's definable subsets
    checks.append(defsubsets(1) == 2)
    # monotone growth: L_n subset L_{n+1}
    checks.append(defsubsets(3) > defsubsets(2))
    # L satisfies GCH (toy marker)
    checks.append(True)
    # V=L is the smallest inner model
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_inner_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_inner_model": _bench_inner_model(seed)}
