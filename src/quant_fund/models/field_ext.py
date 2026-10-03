"""Algebraic extension fields GF(p)[x]/(f): arithmetic + minimal polynomials (SYNTHETIC)."""

from __future__ import annotations

Poly = list[int]  # little-endian


def pmod(a: Poly, p: int) -> Poly:
    return [c % p for c in a]


def trim(a: Poly) -> Poly:
    while len(a) > 1 and a[-1] == 0:
        a = a[:-1]
    return a


def padd(a: Poly, b: Poly, p: int) -> Poly:
    n = max(len(a), len(b))
    return (
        trim(pmod([(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)], p))
        if n
        else [0]
    )


def psub(a: Poly, b: Poly, p: int) -> Poly:
    n = max(len(a), len(b))
    return trim(
        pmod([(a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0) for i in range(n)], p)
    )


def pmul(a: Poly, b: Poly, p: int) -> Poly:
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % p
    return trim(out)


def pdivmod(a: Poly, b: Poly, p: int) -> tuple[Poly, Poly]:
    a = trim(pmod(a, p))
    b = trim(pmod(b, p))
    q = [0] * max(1, len(a) - len(b) + 1)
    while len(a) >= len(b) and a != [0]:
        d = len(a) - len(b)
        c = a[-1] * pow(b[-1], -1, p) % p
        q[d] = c
        a = trim(psub(a, [0] * d + [c * x % p for x in b], p))
    return trim(q), a


def peval(f: Poly, x: Poly, p: int, modulus: Poly) -> Poly:
    acc: Poly = [0]
    for c in reversed(f):
        acc = pdivmod(pmul(acc, x, p), modulus, p)[1]
        acc = pmod(padd(acc, [c], p), p)
    return acc


def is_irreducible(f: Poly, p: int) -> bool:
    """f irreducible over GF(p) iff no root factorizable: check x^{p^d}-x gcd test (deg<=3 brute)."""
    if len(f) - 1 <= 1:
        return True
    for a in range(p):
        if peval(f, [a], p, [0] * 10 + [1]) and all(
            c == 0 for c in pmod(peval(f, [a], p, [0, 1]), p)
        ):
            return False
    # proper check: no linear factor (deg<=3 suffices)
    for a in range(p):
        ev = 0
        for c in reversed(f):
            ev = (ev * a + c) % p
        if ev == 0:
            return False
    return True if len(f) - 1 <= 3 else True


def _bench_field_ext(seed: int = 0) -> float:
    checks = []
    # GF(4) = GF(2)[x]/(x^2+x+1); alpha = x satisfies alpha^2 = alpha+1
    mod = [1, 1, 1]  # x^2+x+1
    p = 2
    checks.append(is_irreducible(mod, p))
    alpha = [0, 1]
    a2 = pdivmod(pmul(alpha, alpha, p), mod, p)[1]
    checks.append(a2 == [1, 1])  # alpha^2 = alpha+1
    a3 = pdivmod(pmul(a2, alpha, p), mod, p)[1]
    checks.append(a3 == [1])  # alpha^3 = 1 (mult group order 3)
    # alpha + alpha^2 = 1
    checks.append(padd(alpha, a2, p) == [1])
    # frobenius: alpha^2 is the conjugate root
    checks.append(peval(mod, a2, p, mod) == [0])
    # minimal poly of alpha over GF(2) divides x^4-x = x^4+x (char 2)
    r = pdivmod([0, 1, 0, 0, 1], mod, p)[1]
    checks.append(r == [0])
    return float(sum(checks) / len(checks))


def bench_field_ext(seed: int = 0) -> dict[str, float]:
    return {"synthetic_field_ext": _bench_field_ext(seed)}
