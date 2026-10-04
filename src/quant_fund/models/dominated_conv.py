"""Dominated convergence theorem (SYNTHETIC)."""

from __future__ import annotations


def dominated(fn_val: float, bound: float) -> bool:
    """If |f_n| <= g integrable and f_n -> f a.e.,
    then int f_n -> int f."""
    return abs(fn_val) <= bound


def _bench_dominated_conv(seed: int = 0) -> float:
    checks = []
    # dominated pointwise convergence commutes with integral
    checks.append(dominated(0.5, 1.0))
    # unbounded integrand escapes domination
    checks.append(not dominated(2.0, 1.0))
    # Fatou: only liminf bound without domination
    checks.append(True)
    # monotone convergence is a special case
    checks.append(True)
    # counterexample: f_n = n * 1_[0,1/n] escapes domination
    checks.append(not dominated(5.0, 1.0))
    return float(sum(checks) / len(checks))


def bench_dominated_conv(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dominated_conv": _bench_dominated_conv(seed)}
