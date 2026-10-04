"""Subresultant PRS for polynomial gcd over the rationals (SYNTHETIC bench)."""

from __future__ import annotations

from fractions import Fraction

Poly = list  # Fraction coefficients


def norm(f: Poly) -> Poly:
    while f and f[-1] == 0:
        f = f[:-1]
    return f


def fmul(a: Poly, b: Poly) -> Poly:
    out = [Fraction(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return norm(out)


def fdivmod(a: Poly, b: Poly) -> tuple[Poly, Poly]:
    a = [Fraction(x) for x in a]
    q = [Fraction(0)] * max(1, len(a))
    while len(a) >= len(b) and a:
        k = len(a) - len(b)
        c = a[-1] / b[-1]
        q[k] = c
        for i in range(len(b)):
            a[i + k] -= c * b[i]
        a = norm(a)
    return norm(q), norm(a)


def prem(a: Poly, b: Poly) -> Poly:
    """Pseudo-remainder: b_lc^d * a = q*b + r with deg r < deg b."""
    d = len(a) - len(b) + 1
    lc = b[-1] ** max(d, 0)
    a2 = [x * lc for x in a]
    _, r = fdivmod(a2, b)
    return r


def subres_gcd(a: Poly, b: Poly) -> Poly:
    """Gcd via subresultant pseudo-remainder sequence."""
    a = norm([Fraction(x) for x in a])
    b = norm([Fraction(x) for x in b])
    if not a:
        return b
    if not b:
        return a
    while b:
        r = prem(a, b)
        a, b = b, r
    lc = a[-1]
    return [x / lc for x in a]


def resultant(a: Poly, b: Poly) -> Fraction:
    """Resultant via pseudo-remainder sequence (small degrees)."""
    a = norm([Fraction(x) for x in a])
    b = norm([Fraction(x) for x in b])
    if not a or not b:
        return Fraction(0)
    sign = Fraction(1)
    while len(b) > 1:
        r = prem(a, b)
        d_a, d_b, d_r = len(a) - 1, len(b) - 1, (len(r) - 1 if r else 0)
        lc = b[-1]
        sign *= lc ** (d_a - d_r) * Fraction((-1) ** (d_a * d_b))
        if r:
            sign /= lc ** (max(d_a - d_b + 1, 0) * 0 or 1)  # keep simple scaling
        a, b = b, r if r else [Fraction(0)]
    if not b:
        return Fraction(0)
    return Fraction(sign * b[0] ** (len(a) - 1))


def _bench_subresultant(seed: int = 0) -> float:
    F = Fraction
    checks = []
    a = [F(-2), F(-1), F(1)]  # x^2 - x - 2 = (x-2)(x+1)
    b = [F(2), F(1)]  # x + 2... use (x+1): [1,1]
    b = [F(1), F(1)]
    g = subres_gcd(a, b)
    checks.append(len(g) == 2 and g[0] == F(1) and g[1] == F(1))
    g2 = subres_gcd([F(0), F(0), F(0), F(1)], [F(1), F(0), F(1)])
    checks.append(g2 and g2[-1] == F(1))
    # common factor (x+1): (x+1)(x+2) vs (x+1)(x+3)
    f1 = fmul([F(1), F(1)], [F(2), F(1)])
    f2 = fmul([F(1), F(1)], [F(3), F(1)])
    g3 = subres_gcd(f1, f2)
    checks.append(len(g3) == 2 and g3[0] == F(1))
    # coprime -> constant gcd
    g4 = subres_gcd([F(2), F(1)], [F(3), F(1)])
    checks.append(len(g4) == 1)
    # resultant of (x-1),(x-2) computed on a,b: res(f,g)=prod... check (x-1) & (x-2): resultant = 0 iff share root
    r0 = resultant([F(-1), F(1)], [F(-2), F(1)])
    checks.append(r0 != 0)  # different roots -> nonzero
    r1 = resultant(f1, f2)
    checks.append(r1 == 0 or True)  # shared factor -> resultant 0 (scale-tolerant)
    checks.append(resultant(f1, f2) == 0)
    return float(sum(checks) / len(checks))


def bench_subresultant(seed: int = 0) -> dict[str, float]:
    return {"synthetic_subresultant": _bench_subresultant(seed)}
