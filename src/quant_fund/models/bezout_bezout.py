"""Bézout intersection numbers for univariate polys over GF(p) (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.field_ext import pdivmod


def intersection_mult(f: list[int], g: list[int], a: int, p: int) -> int:
    """Multiplicity of a as common root: largest k with (x-a)^k | gcd(f,g) at a."""
    m = _pgcd(f, g, p)
    mult = 0
    while len(m) > 1 and _eval_at(m, a, p) == 0:
        m = _pdiv_by_linear(m, a, p)
        mult += 1
    return mult


def _eval_at(f: list[int], a: int, p: int) -> int:
    acc = 0
    for i, c in enumerate(f):
        acc = (acc + c * pow(a, i, p)) % p
    return acc


def _pgcd(f: list[int], g: list[int], p: int) -> list[int]:
    while len(g) > 1 or (len(g) == 1 and g[0] != 0):
        _, r = pdivmod(f, g, p)
        f, g = g, r
        if r == [0]:
            break
    # monic
    if len(f) > 1 or f[0] != 0:
        lead = f[-1]
        inv = pow(lead % p, -1, p)
        f = [(c * inv) % p for c in f]
    return f


def _pdiv_by_linear(f: list[int], a: int, p: int) -> list[int]:
    """Divide f by (x - a) exactly."""
    out = [0] * (len(f) - 1)
    cur = 0
    for i in range(len(f) - 1, 0, -1):
        cur = (f[i] + cur * a) % p
        out[i - 1] = cur
    return out


def _bench_bezout_bezout(seed: int = 0) -> float:
    checks = []
    p = 7
    # f = x^2 - 1 = (x-1)(x+1); g = x - 1: intersection mult at x=1 is 1
    checks.append(intersection_mult([6, 0, 1], [6, 1], 1, p) == 1)
    # f = (x-2)^2; g = (x-2)^2: common factor multiplicity 2 at x=2
    checks.append(intersection_mult([4, 3, 1], [4, 3, 1], 2, p) == 2)
    # coprime: 0
    checks.append(intersection_mult([1, 0, 1], [1, 1], 0, p) == 0)
    # f = x^3 - x = x(x-1)(x+1); g = x^2: mult 1 at 0
    checks.append(intersection_mult([0, 6, 0, 1], [0, 0, 1], 0, p) == 1)
    # bezout counting: deg(f)*deg(g) vs total mult on x^2-x vs x: gcd = x -> mult 1 at 0
    checks.append(intersection_mult([0, 6, 1], [0, 1], 0, p) == 1)
    return float(sum(checks) / len(checks))


def bench_bezout_bezout(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bezout_bezout": _bench_bezout_bezout(seed)}
