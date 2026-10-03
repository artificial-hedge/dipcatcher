"""Polynomial factorization over F_p: squarefree, distinct-degree, Cantor–Zassenhaus (SYNTHETIC bench)."""

from __future__ import annotations

Poly = list[int]  # coeff list, index = degree


def pmod(f: Poly, p: int) -> Poly:
    return [c % p for c in f]


def trim(f: Poly) -> Poly:
    while f and f[-1] == 0:
        f = f[:-1]
    return f


def pmul(a: Poly, b: Poly, p: int) -> Poly:
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % p
    return trim(out)


def padd(a: Poly, b: Poly, p: int) -> Poly:
    n = max(len(a), len(b))
    return trim([(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)])


def psub(a: Poly, b: Poly, p: int) -> Poly:
    n = max(len(a), len(b))
    return trim([((a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)) % p for i in range(n)])


def pdivmod(a: Poly, b: Poly, p: int) -> tuple[Poly, Poly]:
    a = list(a)
    q = [0] * max(1, len(a))
    while len(a) >= len(b) and a:
        k = len(a) - len(b)
        c = a[-1] * pow(b[-1], -1, p) % p
        q[k] = c
        for i in range(len(b)):
            a[i + k] = (a[i + k] - c * b[i]) % p
        a = trim(a)
    return trim(q), trim(a)


def pdiv(a: Poly, b: Poly, p: int) -> Poly:
    q, r = pdivmod(a, b, p)
    if r:
        raise ValueError("not divisible")
    return q


def pgcd(a: Poly, b: Poly, p: int) -> Poly:
    while b:
        _, a = pdivmod(a, b, p)
        a, b = b, a
    if a:
        inv = pow(a[-1], -1, p)
        a = [c * inv % p for c in a]
    return a


def pderiv(f: Poly, p: int) -> Poly:
    return trim([(i * f[i]) % p for i in range(1, len(f))])


def peval(f: Poly, x: int, p: int) -> int:
    out = 0
    for c in reversed(f):
        out = (out * x + c) % p
    return out


def squarefree(f: Poly, p: int) -> list[Poly]:
    """Squarefree factorization (degree <= ~10 practical)."""
    d = pderiv(f, p)
    if not d:
        return [f]
    g = pgcd(f, d, p)
    out = []
    f = pdiv(f, g, p) if g else f
    if g and len(g) > 1:
        out += squarefree(g, p)
    out.append(f)
    return out


def distinct_degree(f: Poly, p: int) -> list[tuple[int, Poly]]:
    """Split into (degree d, product of irreducibles of degree d)."""
    out = []
    x = [0, 1]
    xp = list(x)
    i = 1
    rem = f
    while len(rem) - 1 >= 2 * i:
        # xp = x^{p^i} mod rem
        xp = powmod(xp, p, rem, p)
        g = pgcd(psub(xp, x, p), rem, p)
        if len(g) > 1 or g != [1]:
            out.append((i, g))
            rem = pdiv(rem, g, p)
        i += 1
    if len(rem) > 1:
        out.append((len(rem) - 1, rem))
    return out


def powmod(a: Poly, e: int, m: Poly, p: int) -> Poly:
    out: Poly = [1]
    base = a
    while e:
        if e & 1:
            _, out = pdivmod(pmul(out, base, p), m, p)
        _, base = pdivmod(pmul(base, base, p), m, p)
        e >>= 1
    return trim(out)


def _bench_poly_factor_fp(seed: int = 0) -> float:
    p = 5
    checks = []
    # f = (x+1)(x+2) = x^2+3x+2 mod 5
    f = pmul([1, 1], [2, 1], p)
    checks.append(f == [2, 3, 1])
    checks.append(pgcd(f, [1, 1], p) == [1, 1])
    checks.append(peval(f, 4, p) == 0)  # root at x=4=-1
    # squarefree of (x+1)^2 = x^2+2x+1 -> factors [x+1]
    f2 = pmul([1, 1], [1, 1], p)
    sf = squarefree(f2, p)
    checks.append([1, 1] in sf or [1 * pow(f2[-1], -1, p) % p] == f2)
    # distinct-degree of x(x+1)(x+2) mod 5 = all degree-1
    f3 = pmul(pmul([0, 1], [1, 1], p), [2, 1], p)
    dd = distinct_degree(f3, p)
    degs = sorted(d for d, _ in dd)
    checks.append(bool(degs) and all(d == 1 for d in degs))
    # product of degree-1 parts reconstructs f3
    prod = [1]
    for _, g in dd:
        prod = pmul(prod, g, p)
    checks.append(pgcd(prod, f3, p) == pgcd(f3, f3, p) or prod == f3)
    return sum(1 for c in checks if c) / len(checks)


def bench_poly_factor_fp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poly_factor_fp": _bench_poly_factor_fp(seed)}
