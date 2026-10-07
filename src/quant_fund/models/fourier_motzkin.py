"""Fourier–Motzkin elimination over Q (SYNTHETIC).

A constraint system is a list of rows (a, b) meaning a·x <= b with
exact Fraction arithmetic. eliminate(var) produces the projection onto
the remaining variables; feasible(sys) decides existence of a rational
point — exact, no LP solver needed.
"""

from __future__ import annotations

from fractions import Fraction

_SEED = 20261231 + 1029

Row = tuple[list[Fraction], Fraction]
Sys = list[Row]


def feasible(sys_: Sys) -> bool:
    sys_ = [([Fraction(c) for c in a], Fraction(b)) for a, b in sys_]
    if not sys_:
        return True
    n = len(sys_[0][0])
    for _ in range(n):
        sys_ = eliminate(sys_, 0)
    return all(b >= 0 for _a, b in sys_)


def _was_infeasible(s: Sys) -> bool:
    return any(all(c == 0 for c in a) and b < 0 for a, b in s)


def eliminate(sys_: Sys, v: int) -> Sys:
    """Eliminate variable v: combine every upper bound with every lower
    bound; keep bounds that don't mention v."""
    lo: list[Row] = []
    hi: list[Row] = []
    rest: list[Row] = []
    for a, b in sys_:
        c = a[v]
        if c > 0:  # c*x <= rest -> x <= (b - others)/c  (upper bound on x)
            hi.append((a, b))
        elif c < 0:  # lower bound on x
            lo.append((a, b))
        else:
            rest.append((a, b))
    out: Sys = [([c for i, c in enumerate(a) if i != v], b) for a, b in rest]
    for la, lb in lo:
        for ha, hb in hi:
            # la[v]*x <= lb - la_rest  =>  x >= (lb - la_rest)/la[v] (la[v]<0)
            # ha[v]*x <= hb - ha_rest  =>  x <= (hb - ha_rest)/ha[v]
            # require (lb - la_rest)/la[v] <= (hb - ha_rest)/ha[v]
            lc, hc = la[v], ha[v]
            new_a: list[Fraction] = []
            new_b = hc * lb - lc * hb
            for i, (lc_, hc_) in enumerate(zip(la, ha, strict=True)):
                if i == v:
                    continue
                new_a.append(hc * lc_ - lc * hc_)
            out.append((new_a, new_b))
    # drop trivially-false rows: all-zero lhs with negative rhs -> infeasible marker
    for a, b in out:
        if all(c == 0 for c in a) and b < 0:
            return [([Fraction(0)], Fraction(-1))]
    return out


def bench_fourier_motzkin(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []
    F = Fraction
    # triangle: x>=0 (-x<=0), y>=0, x+y<=1 — feasible
    tri: Sys = [([-F(1), F(0)], F(0)), ([F(0), -F(1)], F(0)), ([F(1), F(1)], F(1))]
    checks.append(feasible(tri))
    # x >= 2 and x <= 1 — infeasible
    bad: Sys = [([-F(1)], -F(2)), ([F(1)], F(1))]
    checks.append(not feasible(bad))
    # projection: eliminate y from {x+y<=4, y>=1, x>=0} -> x<=3, x>=0 stays feasible
    s2: Sys = [([F(1), F(1)], F(4)), ([F(0), -F(1)], -F(1)), ([-F(1), F(0)], F(0))]
    proj = eliminate(s2, 1)
    checks.append(any(b == 3 for a, b in proj if a == [F(1)]) or feasible(proj))
    # classic: x+y<=2, x-y>=0, x>=1 -> y<=1/2 exists (x=1,y=0)
    s3: Sys = [
        ([F(1), F(1)], F(2)),
        ([-F(1), F(1)], F(0)),
        ([-F(1), F(0)], -F(1)),
    ]
    checks.append(feasible(s3))
    # y >= 3 added -> infeasible
    s4 = s3 + [([F(0), -F(1)], -F(3))]
    checks.append(not feasible(s4))
    return {"synthetic_fourier_motzkin": float(sum(checks)) / len(checks)}
