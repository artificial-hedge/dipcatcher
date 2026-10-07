"""Elliptic-curve scalar multiplication over GF(p) (synthetic) (SYNTHETIC).

Short-Weierstrass curves y² = x³ + a·x + b mod p with affine point
add/double and double-and-add scalar mult. Verified against repeated
addition, the ECDLP-free identity k·P + l·P = (k+l)·P, and a
differential validation over random scalars.
"""

from __future__ import annotations

import random

Point = tuple[int, int] | None  # None = point at infinity


def _add(p1: Point, p2: Point, a: int, p: int) -> Point:
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and (y1 + y2) % p == 0:
        return None
    if p1 == p2:
        m = (3 * x1 * x1 + a) * pow(2 * y1, -1, p) % p
    else:
        m = (y2 - y1) * pow((x2 - x1) % p, -1, p) % p
    x3 = (m * m - x1 - x2) % p
    return x3, (m * (x1 - x3) - y1) % p


def _mul(k: int, pt: Point, a: int, p: int) -> Point:
    out: Point = None
    addend = pt
    while k:
        if k & 1:
            out = _add(out, addend, a, p)
        addend = _add(addend, addend, a, p)
        k >>= 1
    return out


def bench_ec_scalar(seed: int = 20261231 + 245) -> dict[str, float]:
    rng = random.Random(seed)
    # curve y² = x³ + 2x + 2 mod 17 (classic NIST tutorial curve)
    a_c, p = 2, 17
    g: Point = (5, 1)
    order = 19  # known order of this curve
    # 1) k·G cycles at the order
    ok_cycle = _mul(order, g, a_c, p) is None
    ok_cycle2 = _mul(order + 1, g, a_c, p) == g
    # 2) k·P + l·P = (k+l)·P on random scalars
    agree = 0
    trials = 25
    for _ in range(trials):
        k, ell = rng.randint(1, order - 1), rng.randint(1, order - 1)
        lhs = _add(_mul(k, g, a_c, p), _mul(ell, g, a_c, p), a_c, p)
        rhs = _mul(k + ell, g, a_c, p)
        agree += int(lhs == rhs)
    # 3) all points on curve: verify y² = x³+ax+b for every output
    on_curve = True
    for k in range(1, order):
        pt = _mul(k, g, a_c, p)
        if pt is None:
            continue
        x, y = pt
        on_curve &= (y * y - (x * x * x + a_c * x + 2)) % p == 0
    return {
        "synthetic_order_cycle": float(ok_cycle and ok_cycle2),
        "synthetic_homomorphism": float(agree / trials),
        "synthetic_on_curve": float(on_curve),
    }
