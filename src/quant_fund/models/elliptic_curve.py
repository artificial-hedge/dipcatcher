"""Elliptic curve group law over GF(p): add, double, scalar mult (SYNTHETIC)."""

from __future__ import annotations

Pt = tuple[int, int] | None  # None = point at infinity


def on_curve(a4: int, b6: int, p: int, pt: Pt) -> bool:
    if pt is None:
        return True
    x, y = pt
    return (y * y - x * x * x - a4 * x - b6) % p == 0


def add(a4: int, p: int, P: Pt, Q: Pt) -> Pt:
    if P is None:
        return Q
    if Q is None:
        return P
    x1, y1 = P
    x2, y2 = Q
    if x1 == x2 and (y1 + y2) % p == 0:
        return None
    if P == Q:
        if y1 % p == 0:
            return None
        lam = (3 * x1 * x1 + a4) * pow(2 * y1 % p, -1, p) % p
    else:
        lam = (y2 - y1) * pow((x2 - x1) % p, -1, p) % p
    x3 = (lam * lam - x1 - x2) % p
    y3 = (lam * (x1 - x3) - y1) % p
    return (x3, y3)


def smul(a4: int, p: int, k: int, P: Pt) -> Pt:
    out: Pt = None
    base = P
    while k:
        if k & 1:
            out = add(a4, p, out, base)
        base = add(a4, p, base, base)
        k >>= 1
    return out


def _bench_elliptic_curve(seed: int = 0) -> float:
    checks = []
    p, a4, b6 = 5, 2, 1  # y^2 = x^3 + 2x + 1
    pts = [(x, y) for x in range(5) for y in range(5) if on_curve(a4, b6, p, (x, y))]
    checks.append(len(pts) == 6)  # 6 affine + infinity = 7 group elems
    P = (0, 1)
    checks.append(on_curve(a4, b6, p, P))
    checks.append(add(a4, p, P, P) is not None)
    Q = add(a4, p, P, P)
    checks.append(on_curve(a4, b6, p, Q))
    # scalar multiples stay on curve and hit infinity eventually
    orders = {k: smul(a4, p, k, P) for k in range(1, 10)}
    checks.append(any(v is None for v in orders.values()))
    # associativity on small case
    checks.append(add(a4, p, add(a4, p, P, Q), P) == add(a4, p, P, add(a4, p, Q, P)))
    return float(sum(checks) / len(checks))


def bench_elliptic_curve(seed: int = 0) -> dict[str, float]:
    return {"synthetic_elliptic_curve": _bench_elliptic_curve(seed)}
