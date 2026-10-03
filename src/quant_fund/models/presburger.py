"""Presburger-style bounded linear-integer satisfiability (SYNTHETIC)."""

from __future__ import annotations


def sat_exists_y(x: int, a: int, b: int, lo: int, hi: int) -> bool:
    """exists y in [lo,hi]: x + y in [a,b]."""
    return any(a <= x + y <= b for y in range(lo, hi + 1))


def _bench_presburger(seed: int = 0) -> float:
    checks = []
    # forall x in [0,4] exists y: x+y = 4 -> decidable true on domain 0..4
    checks.append(all(sat_exists_y(x, 4, 4, 0, 4) for x in range(5)))
    # exists x forall y: x+y <= 5 on y in 0..3 -> x = 0,1,2 works
    checks.append(any(all(x + y <= 5 for y in range(4)) for x in range(4)))
    # forall x exists y: x + y >= x (y >= 0) trivially true on naturals
    checks.append(all(sat_exists_y(x, x, x + 10, 0, 4) for x in range(5)))
    # unsat: exists x: x + x = 3 has no integer solution
    checks.append(not any(2 * x == 3 for x in range(10)))
    # but x + x = 4 does
    checks.append(any(2 * x == 4 for x in range(10)))
    # quantifier alternation: forall x exists y: 2y >= x over 0..9
    checks.append(all(any(2 * y >= x for y in range(10)) for x in range(10)))
    # false version: forall x exists y: 2y = x + 1 fails at x even
    checks.append(not all(any(2 * y == x + 1 for y in range(10)) for x in range(10)))
    return float(sum(checks) / len(checks))


def bench_presburger(seed: int = 0) -> dict[str, float]:
    return {"synthetic_presburger": _bench_presburger(seed)}
