"""Hensel lifting: lift mod-p factorization to mod p^k (SYNTHETIC bench)."""

from __future__ import annotations


def zmod(f: list[int], m: int) -> list[int]:
    out = [c % m for c in f]
    while out and out[-1] == 0:
        out.pop()
    return out


def zmul(a: list[int], b: list[int], m: int) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return zmod(out, m)


def zsub(a: list[int], b: list[int], m: int) -> list[int]:
    n = max(len(a), len(b))
    return zmod([(a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0) for i in range(n)], m)


def lift(f: list[int], g: list[int], h: list[int], p: int, k: int) -> tuple[list[int], list[int]]:
    """Lift f ≡ g·h (mod p) to g·h ≡ f (mod p^k), coprime g,h mod p."""
    g, h = zmod(g, p), zmod(h, p)
    m = p
    while m < p**k:
        m2 = m * p
        # e = (f - g*h)/m mod p; find s,t with s*g + t*h ≡ e mod p
        e = zsub(f, zmul(g, h, m2), m2)
        e = zmod([c // m for c in e], p)
        s, t = bezout(g, h, e, p)
        g = zmod([g[i] + m * (s[i] if i < len(s) else 0) for i in range(max(len(g), len(s)))], m2)
        h = zmod([h[i] + m * (t[i] if i < len(t) else 0) for i in range(max(len(h), len(t)))], m2)
        m = m2
    return g, h


def bezout(g: list[int], h: list[int], e: list[int], p: int) -> tuple[list[int], list[int]]:
    """Find s,t with s*g + t*h ≡ e (mod p) via extended gcd on h,g then scale."""
    # egcd: a*u + b*v = gcd; over F_p with gcd=1 (assumed coprime)
    a, b = list(h), list(g)
    u1, u2 = [1], [0]
    v1, v2 = [0], [1]
    while b:
        q, r = pdivmod_e(a, b, p)
        a, b = b, r
        u1, u2 = u2, zsub(u1, zmul(q, u2, p), p)
        v1, v2 = v2, zsub(v1, zmul(q, v2, p), p)
    # u1*h + v1*g = 1.  Corrections: g += m*(e*u1 mod h), h += m*(e*v1 mod g)
    g_corr = zmul(e, u1, p)
    h_corr = zmul(e, v1, p)
    _, g_corr = pdivmod_e(g_corr, h, p)
    _, h_corr = pdivmod_e(h_corr, g, p)
    return g_corr, h_corr


def pdivmod_e(a: list[int], b: list[int], m: int) -> tuple[list[int], list[int]]:
    a = list(a)
    q = [0] * max(1, len(a))
    while len(a) >= len(b) and a:
        k = len(a) - len(b)
        c = a[-1] * pow(b[-1], -1, m) % m
        q[k] = c
        for i in range(len(b)):
            a[i + k] = (a[i + k] - c * b[i]) % m
        while a and a[-1] == 0:
            a.pop()
    return [x % m for x in q], a


def _bench_hensel_lift(seed: int = 0) -> float:
    p, k = 5, 2
    checks = []
    # f = x^2 + 3x + 2 = (x+1)(x+2) mod 5; lift to mod 25
    f = [2, 3, 1]
    g, h = [1, 1], [2, 1]
    g2, h2 = lift(f, g, h, p, k)
    prod = zmul(g2, h2, p**k)
    checks.append(prod == zmod(f, p**k))
    checks.append(zmod(g2, p) == g and zmod(h2, p) == h)
    # lift a quadratic root: x^2 - 2 mod 7 has root 3; lift via derivative map
    # use poly form: (x-3)(x-4) = x^2+0x+5 mod 7
    f2 = [12, -7, 1]  # x^2 - 7x + 12 = (x-3)(x-4) over Z
    g3, h3 = lift(f2, zmod([-3, 1], 7), zmod([-4, 1], 7), 7, 2)
    prod2 = zmul(g3, h3, 49)
    checks.append(prod2 == zmod(f2, 49))
    return sum(checks) / len(checks)


def bench_hensel_lift(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hensel_lift": _bench_hensel_lift(seed)}
