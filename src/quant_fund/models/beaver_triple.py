"""Beaver multiplication triples on additive shares (SYNTHETIC bench)."""

from __future__ import annotations

import random


def split(v: int, n: int, p: int, rng: random.Random) -> list[int]:
    """Additive secret sharing."""
    parts = [rng.randrange(p) for _ in range(n - 1)]
    parts.append((v - sum(parts)) % p)
    return parts


def triple(p: int, rng: random.Random) -> tuple[list[int], list[int], list[int]]:
    """Beaver triple (a, b, c=ab) additively shared."""
    a, b = rng.randrange(p), rng.randrange(p)
    return split(a, 2, p, rng), split(b, 2, p, rng), split(a * b % p, 2, p, rng)


def beaver_mul(
    xs: list[int], ys: list[int], trip: tuple[list[int], list[int], list[int]], p: int
) -> list[int]:
    """Multiply additively shared x,y using triple (a,b,c):
    each party computes d_i = x_i - a_i, e_i = y_i - b_i; open d,e;
    z_i = c_i + d*b_i + e*a_i + (i==0 ? d*e : 0)."""
    (a, b, c) = trip
    n = len(xs)
    d = [(x - ai) % p for x, ai in zip(xs, a, strict=True)]
    e = [(y - bi) % p for y, bi in zip(ys, b, strict=True)]
    d_pub = sum(d) % p
    e_pub = sum(e) % p
    z = [
        (c[i] + d_pub * b[i] + e_pub * a[i] + (d_pub * e_pub if i == 0 else 0)) % p
        for i in range(n)
    ]
    return z


def _bench_beaver_triple(seed: int = 0) -> float:
    rng = random.Random(20261231 + 1912 + seed)
    p = 211
    checks = []
    x = split(17, 2, p, rng)
    checks.append(sum(x) % p == 17)
    for a, b in ((3, 4), (50, 60), (0, 7)):
        xs = split(a, 2, p, rng)
        ys = split(b, 2, p, rng)
        z = beaver_mul(xs, ys, triple(p, rng), p)
        checks.append(sum(z) % p == a * b % p)
    # triple correctness
    ta, tb, tc = triple(p, rng)
    checks.append(sum(ta) % p * (sum(tb) % p) % p == sum(tc) % p)
    return sum(checks) / len(checks)


def bench_beaver_triple(seed: int = 0) -> dict[str, float]:
    return {"synthetic_beaver_triple": _bench_beaver_triple(seed)}
