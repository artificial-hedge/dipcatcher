"""Splitting fields: full factorization in extension towers (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.field_ext import pdivmod, pmod, pmul, trim

Poly = list[int]


def roots_in(f: Poly, p: int, mod: Poly | None = None) -> list[Poly]:
    """Roots of f over GF(p) (mod=None) or extension GF(p)[a]/mod — brute elements."""
    if mod is None:
        return [[a] for a in range(p) if sum(c * a**i for i, c in enumerate(f)) % p == 0]
    deg = len(mod) - 1
    elems = []
    for mask in range(p**deg):
        e = []
        m = mask
        for _ in range(deg):
            e.append(m % p)
            m //= p
        elems.append(trim(e))

    def ev(x: Poly) -> Poly:
        acc: Poly = [0]
        for c in reversed(f):
            acc = pdivmod(pmul(acc, x, p), mod, p)[1]
            acc = pmod(
                [
                    (acc[i] if i < len(acc) else 0) + (c if i == 0 else 0)
                    for i in range(max(len(acc), 1))
                ],
                p,
            )
        return trim(acc)

    return [e for e in elems if ev(e) == [0]]


def splits_over(f: Poly, p: int, mod: Poly | None = None) -> bool:
    deg = len(f) - 1
    return len(roots_in(f, p, mod)) == deg


def _bench_splitting_field(seed: int = 0) -> float:
    checks = []
    # x^3 - 2 over GF(5): has no root? check 3^3=27=2 -> x=3 root! Use x^2+1 instead.
    f = [1, 0, 1]  # x^2 + 1: roots mod 5 are 2,3 -> splits over GF(5) already
    checks.append(splits_over(f, 5))
    # x^2 - 2 over GF(5): roots? 3^2=9=4,4^2=16=1, no -> irreducible; adjoin alpha
    g = [3, 0, 1]  # x^2 - 2 = x^2 + 3
    checks.append(not splits_over(g, 5))
    # over GF(25)=GF(5)[a]/(a^2+a+1)... pick a valid quadratic irr: a^2+a+1? check roots mod5: f(0)=1,f(1)=3,f(2)=2,f(3)=3,f(4)=1 none -> irreducible
    mod = [1, 1, 1]
    roots = roots_in(g, 5, mod)
    checks.append(len(roots) == 2)  # g splits in GF(25) (every quadratic splits in GF(p^2))
    # x^2+x+1 itself splits over GF(25)? degree 2 poly over GF(25): its roots live in GF(5^2)/GF(5) iff discr square... it's irreducible over GF5 so roots are in GF25
    checks.append(len(roots_in(mod, 5, mod)) == 2)
    return float(sum(checks) / len(checks))


def bench_splitting_field(seed: int = 0) -> dict[str, float]:
    return {"synthetic_splitting_field": _bench_splitting_field(seed)}
