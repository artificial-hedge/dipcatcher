"""Commutative-ring decision procedure via polynomial normal form (SYNTHETIC).

Ring expressions over vars (strings) and int literals normalize to a
coefficient map {sorted-variable-tuple monomial: coefficient}; ring equality
is coefficient-map equality. This is the `ring` tactic's core: decide
commutative ring identities by normalization rather than rewriting.
"""

from __future__ import annotations

from typing import Any

_SEED = 20261231 + 1000

Expr = Any
Poly = dict[tuple[str, ...], int]

_ZERO: Poly = {}
_ONE: Poly = {(): 1}


def _padd(p: Poly, q: Poly) -> Poly:
    out = dict(p)
    for m, c in q.items():
        out[m] = out.get(m, 0) + c
    return {m: c for m, c in out.items() if c != 0}


def _pmul(p: Poly, q: Poly) -> Poly:
    out: Poly = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            m = tuple(sorted(m1 + m2))
            out[m] = out.get(m, 0) + c1 * c2
    return {m: c for m, c in out.items() if c != 0}


def _pneg(p: Poly) -> Poly:
    return {m: -c for m, c in p.items()}


def normalize(e: Expr) -> Poly:
    tag = e[0]
    if tag == "lit":
        return {(): int(e[1])}
    if tag == "var":
        return {(str(e[1]),): 1}
    if tag == "add":
        return _padd(normalize(e[1]), normalize(e[2]))
    if tag == "mul":
        return _pmul(normalize(e[1]), normalize(e[2]))
    if tag == "neg":
        return _pneg(normalize(e[1]))
    if tag == "sub":
        return _padd(normalize(e[1]), _pneg(normalize(e[2])))
    if tag == "pow":
        base = normalize(e[1])
        out = dict(_ONE)
        for _ in range(int(e[2])):
            out = _pmul(out, base)
        return out
    raise ValueError(f"bad ring expr {tag}")


def ring_eq(a: Expr, b: Expr) -> bool:
    return normalize(a) == normalize(b)


def bench_ring_normalize(seed: int = _SEED) -> dict[str, float]:
    """Ring identities decided exactly by normal-form equality."""
    del seed
    x, y = ("var", "x"), ("var", "y")
    two = ("lit", 2)
    xy = ("mul", x, y)
    x2 = ("pow", x, 2)
    y2 = ("pow", y, 2)
    checks: list[bool] = []
    # (x+y)^2 = x^2 + 2xy + y^2
    checks.append(ring_eq(("pow", ("add", x, y), 2), ("add", ("add", x2, y2), ("mul", two, xy))))
    # (x+y)(x-y) = x^2 - y^2
    checks.append(ring_eq(("mul", ("add", x, y), ("sub", x, y)), ("sub", x2, y2)))
    # x*y + x*y = 2*x*y (coefficient merging)
    checks.append(ring_eq(("add", xy, xy), ("mul", two, xy)))
    # x*y != x+y (distinct monomials)
    checks.append(not ring_eq(xy, ("add", x, y)))
    # distributivity: x*(y+y) = 2xy
    checks.append(ring_eq(("mul", x, ("add", y, y)), ("mul", two, xy)))
    return {"synthetic_ring_normalize": float(sum(checks)) / len(checks)}
