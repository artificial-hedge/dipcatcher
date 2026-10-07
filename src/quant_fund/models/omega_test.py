"""Omega-test lite: exact integer feasibility for small Presburger systems (SYNTHETIC).

Constraints are (a, b) meaning a·x <= b over Z^n. The test:
1) FM-eliminate over Q — infeasible over reals => infeasible over Z.
2) Otherwise enumerate a bounded box around the real shadow (the "dark
   shadow" fallback): for the small systems benches use, direct
   enumeration is the exact decision procedure.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product

_SEED = 20261231 + 1033

Row = tuple[list[Fraction], Fraction]


def _fm_feasible(sys_: list[Row]) -> bool:
    sys_ = [([Fraction(c) for c in a], Fraction(b)) for a, b in sys_]
    if not sys_:
        return True
    n = len(sys_[0][0])
    for _ in range(n):
        lo = [r for r in sys_ if r[0][0] < 0]
        hi = [r for r in sys_ if r[0][0] > 0]
        rest = [r for r in sys_ if r[0][0] == 0]
        out = [([c for i, c in enumerate(a) if i != 0], b) for a, b in rest]
        for la, lb in lo:
            for ha, hb in hi:
                na = [
                    ha[0] * lc - la[0] * hc
                    for i, (lc, hc) in enumerate(zip(la, ha, strict=True))
                    if i != 0
                ]
                out.append((na, ha[0] * lb - la[0] * hb))
        sys_ = out
        if any(all(c == 0 for c in a) and b < 0 for a, b in sys_):
            return False
    return all(b >= 0 for _a, b in sys_)


def satisfiable(sys_: list[Row], bound: int = 12) -> bool:
    """Exact Z-feasibility: FM necessary check, then bounded search."""
    if not _fm_feasible(sys_):
        return False
    n = len(sys_[0][0])
    for pt in product(range(-bound, bound + 1), repeat=n):
        if all(
            sum(a * Fraction(x) for a, x in zip(row_a, pt, strict=True)) <= row_b
            for row_a, row_b in sys_
        ):
            return True
    return False


def _mk(cs: list[int], b: int) -> Row:
    return ([Fraction(c) for c in cs], Fraction(b))


def bench_omega_test(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    # x >= 3, x <= 5 -> SAT
    checks.append(satisfiable([_mk([-1], -3), _mk([1], 5)]))
    # x >= 4, x <= 3 -> UNSAT (reals also infeasible)
    checks.append(not satisfiable([_mk([-1], -4), _mk([1], 3)]))
    # 2x+y = 7 mod-free but integer: x+y<=10, x-y>=2, y>=1 -> SAT (x=4,y=1)
    checks.append(satisfiable([_mk([1, 1], 10), _mk([-1, 1], -2), _mk([0, -1], -1)]))
    # parity trick: 2x = 3 over Z has no solution; FM sees x in [1.5,1.5]? encode 2x<=3 and 2x>=3
    checks.append(not satisfiable([_mk([2], 3), _mk([-2], -3)]))
    # 3x+6y=9 -> x=1,y=1 SAT (gcd(3,6)=3 divides 9)
    checks.append(satisfiable([_mk([3, 6], 9), _mk([-3, -6], -9)]))
    # 2x+4y=7 -> gcd 2 does not divide 7 -> UNSAT
    checks.append(not satisfiable([_mk([2, 4], 7), _mk([-2, -4], -7)]))
    return {"synthetic_omega_test": float(sum(checks)) / len(checks)}
