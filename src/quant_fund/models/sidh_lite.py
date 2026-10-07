"""SIDH-lite: toy supersingular isogeny Diffie-Hellman over F_p (SYNTHETIC).

E: y^2 = x^3 + x over F_431 has #E = 432 = 2^4 * 3^3. Alice computes a
2-isogeny chain E -> E/<R_A>; Bob a 3-isogeny chain E -> E/<R_B>. Shared
secret = j-invariant of the composed quotient, equal from both sides.
Velu formulas: t = sum(3x_Q^2 + a), w = sum(5x_Q^3 + 3a x_Q + 2b) over the
nonzero kernel points; a' = a - 5t, b' = b - 7w.
"""

from __future__ import annotations

import numpy as np

_SEED = 20261231 + 943

P = 431
A0, B0 = 1, 0  # y^2 = x^3 + x
INF: tuple[int, int] | None = None


def _inv(x: int, p: int = P) -> int:
    return pow(x, p - 2, p)


def add(
    p1: tuple[int, int] | None, p2: tuple[int, int] | None, a: int = A0, p: int = P
) -> tuple[int, int] | None:
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and (y1 + y2) % p == 0:
        return None
    if p1 == p2:
        lam = (3 * x1 * x1 + a) * _inv(2 * y1, p) % p
    else:
        lam = (y2 - y1) * _inv((x2 - x1) % p, p) % p
    x3 = (lam * lam - x1 - x2) % p
    return x3, (lam * (x1 - x3) - y1) % p


def mul(k: int, pt: tuple[int, int] | None, a: int = A0, p: int = P) -> tuple[int, int] | None:
    r = None
    q = pt
    while k:
        if k & 1:
            r = add(r, q, a, p)
        q = add(q, q, a, p)
        k >>= 1
    return r


def points_on_curve(a: int = A0, b: int = B0, p: int = P) -> list[tuple[int, int]]:
    pts = []
    for x in range(p):
        rhs = (x * x * x + a * x + b) % p
        for y in range(p):
            if y * y % p == rhs:
                pts.append((x, y))
    return pts


def order(pt: tuple[int, int], bound: int = 500, a: int = A0, p: int = P) -> int:
    acc = None
    for k in range(1, bound):
        acc = add(acc, pt, a, p)
        if acc is None:
            return k
    return 0


def j_invariant(a: int, b: int, p: int = P) -> int:
    num = 4 * a * a * a % p
    den = (num + 27 * b * b) % p
    return 1728 * num * _inv(den, p) % p


def velu(kernel: list[tuple[int, int]], a: int, b: int, p: int = P) -> tuple[int, int]:
    """Isogenous curve params for quotient by the given finite subgroup."""
    t = sum(3 * x * x + a for x, _ in kernel) % p
    w = sum(5 * x * x * x + 3 * a * x + 2 * b for x, _ in kernel) % p
    return (a - 5 * t) % p, (b - 7 * w) % p


def isogeny_eval(
    pt: tuple[int, int] | None,
    kernel: list[tuple[int, int]],
    a: int,
    p: int = P,
) -> tuple[int, int] | None:
    """Velu map phi: x' = x + sum(x_{P+Q} - x_Q), y' likewise."""
    if pt is None:
        return None
    x, y = pt
    dx = 0
    dy = 0
    for q in kernel:
        s = add(pt, q, a, p)
        if s is None:
            return None  # pt in kernel -> image is O
        dx = (dx + s[0] - q[0]) % p
        dy = (dy + s[1] - q[1]) % p
    return (x + dx) % p, (y + dy) % p


def _subgroup(r: tuple[int, int], a: int, p: int = P) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    acc = None
    while True:
        acc = add(acc, r, a, p)
        if acc is None:
            return out
        out.append(acc)


def _find_basis(order_target: int, cofactor: int, a: int = A0, p: int = P) -> tuple[int, int]:
    for pt in points_on_curve(a, B0, p):
        q = mul(cofactor, pt, a, p)
        if q is not None and order(q, a=a, p=p) == order_target:
            return q
    raise RuntimeError("no basis point")


def _chain_with_push(
    r: tuple[int, int],
    deg: int,
    e: int,
    push: tuple[int, int],
    a: int = A0,
    b: int = B0,
    p: int = P,
) -> tuple[int, int, tuple[int, int] | None]:
    """Chain quotienting by <r>, returning endpoint curve and image of push."""
    cur_a, cur_b = a, b
    cur: tuple[int, int] | None = r
    img: tuple[int, int] | None = push
    for i in range(e):
        ri = mul(deg ** (e - 1 - i), cur, cur_a, p)
        if not (ri is not None and order(ri, a=cur_a, p=p) == deg):
            raise ValueError("ri is not None and order(ri, a=cur_a, p=p) == deg")
        ker = _subgroup(ri, cur_a, p)
        next_a, next_b = velu(ker, cur_a, cur_b, p)
        cur = isogeny_eval(cur, ker, cur_a, p)
        img = isogeny_eval(img, ker, cur_a, p)
        cur_a, cur_b = next_a, next_b
    return cur_a, cur_b, img


def bench_sidh_lite(seed: int = _SEED) -> dict[str, float]:
    _ = np.random.default_rng(seed)
    checks = []
    pts = points_on_curve(A0, B0, P)
    checks.append(len(pts) + 1 == 432)
    pa = _find_basis(8, 54)
    pb = _find_basis(27, 16)
    checks.append(order(pa) == 8 and order(pb) == 27)
    # Alice: 3-step 2-isogeny chain; Bob: 3-step 3-isogeny chain.
    ea_a, eb_a, phi_a_pb = _chain_with_push(pa, 2, 3, pb)
    ea_b, eb_b, phi_b_pa = _chain_with_push(pb, 3, 3, pa)
    # Alice recomputes shared curve from Bob's curve using image of PA.
    if not (phi_b_pa is not None and phi_a_pb is not None):
        raise ValueError("phi_b_pa is not None and phi_a_pb is not None")
    ea_ab, eb_ab, _ = _chain_with_push(phi_b_pa, 2, 3, phi_b_pa, ea_b, eb_b)
    ea_ba, eb_ba, _ = _chain_with_push(phi_a_pb, 3, 3, phi_a_pb, ea_a, eb_a)
    j1 = j_invariant(ea_ab, eb_ab)
    j2 = j_invariant(ea_ba, eb_ba)
    checks.append(j1 == j2)
    checks.append(phi_a_pb is not None and phi_b_pa is not None)
    if phi_b_pa is not None:
        x, y = phi_b_pa
        checks.append((y * y - x**3 - ea_b * x - eb_b) % P == 0)
    else:
        checks.append(False)
    _, _, phi_a_pa = _chain_with_push(pa, 2, 3, pa)
    checks.append(phi_a_pa is None)  # kernel point lands at O
    return {"synthetic_sidh_lite": float(np.mean(checks))}
