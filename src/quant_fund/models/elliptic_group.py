"""Elliptic curve group law over a prime field (SYNTHETIC)."""

from __future__ import annotations


def inv_mod(a: int, p: int) -> int:
    return pow(a % p, p - 2, p)


def ec_add(P, Q, a: int, p: int):
    """Chord-and-tangent addition on y^2 = x^3 + a*x + b mod p; None = O."""
    if P is None:
        return Q
    if Q is None:
        return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2 and (y1 + y2) % p == 0:
        return None
    if P == Q:
        lam = (3 * x1 * x1 + a) * inv_mod(2 * y1, p) % p
    else:
        lam = (y2 - y1) * inv_mod(x2 - x1, p) % p
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return (x3, y3)


def ec_mul(k: int, P, a: int, p: int):
    r = None
    base = P
    while k:
        if k & 1:
            r = ec_add(r, base, a, p)
        base = ec_add(base, base, a, p)
        k >>= 1
    return r


def points_on_curve(a: int, b: int, p: int) -> list[tuple[int, int]]:
    pts = []
    for x in range(p):
        rhs = (x**3 + a * x + b) % p
        for y in range(p):
            if y * y % p == rhs:
                pts.append((x, y))
    return pts


def _bench_elliptic_group(seed: int = 0) -> float:
    checks = []
    a, b, p = 1, 1, 23
    pts = points_on_curve(a, b, p)
    n = len(pts) + 1  # + point at infinity
    # Hasse bound: |#E - (p+1)| <= 2 sqrt(p)
    checks.append(abs(n - (p + 1)) <= 2 * p**0.5 + 0.01)
    # closure: sums stay on curve or O
    sample = pts[:6]
    closed = True
    for P in sample:
        for Q in sample:
            R = ec_add(P, Q, a, p)
            if R is not None:
                x3, y3 = R
                if (y3 * y3 - (x3**3 + a * x3 + b)) % p != 0:
                    closed = False
    checks.append(closed)
    # inverses: P + (-P) = O
    P = pts[0]
    neg = (P[0], (-P[1]) % p)
    checks.append(ec_add(P, neg, a, p) is None)
    # associativity spot-check
    P, Q, R = pts[0], pts[1], pts[2]
    checks.append(ec_add(ec_add(P, Q, a, p), R, a, p) == ec_add(P, ec_add(Q, R, a, p), a, p))
    # scalar: n*P = O for all P (Lagrange)
    checks.append(all(ec_mul(n, q_, a, p) is None for q_ in sample))
    # doubling formula equals P+P
    checks.append(ec_mul(2, P, a, p) == ec_add(P, P, a, p))
    return float(sum(checks) / len(checks))


def bench_elliptic_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_elliptic_group": _bench_elliptic_group(seed)}
