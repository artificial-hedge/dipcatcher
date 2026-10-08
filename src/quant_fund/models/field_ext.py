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


def _monics(deg: int, p: int) -> list[Poly]:
    """All monic polynomials of degree `deg` over GF(p)."""
    out: list[Poly] = []
    total = p**deg
    for code in range(total):
        coeffs = []
        v = code
        for _ in range(deg):
            coeffs.append(v % p)
            v //= p
        coeffs.append(1)
        out.append(coeffs)
    return out


def is_irreducible(f: Poly, p: int) -> bool:
    """f irreducible over GF(p): no root (kills linear factors), and for
    deg > 3 no monic divisor of degree 2..deg//2 either — a composite like
    (x^2+x+1)^2 over GF(2) has no roots but is reducible."""
    deg = len(f) - 1
    if deg <= 1:
        return True
    if f[-1] % p == 0:
        return False  # not monic after reduction / degenerate lead
    # linear factors: root test
    for a in range(p):
        ev = 0
        for c in reversed(f):
            ev = (ev * a + c) % p
        if ev == 0:
            return False
    if deg <= 3:
        return True  # no linear factor => irreducible
    # higher degrees: trial division by every monic poly of degree 2..deg//2
    for d in range(2, deg // 2 + 1):
        for g in _monics(d, p):
            if pdivmod(f, g, p)[1] == [0]:
                return False
    return True


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
    # degree>3 irreducibility: (x^2+x+1)^2 = x^4+x^2+1 has no roots but
    # is reducible; x^4+x+1 is genuinely irreducible over GF(2)
    sq = pmul(mod, mod, p)
    checks.append(sq == [1, 0, 1, 0, 1])
    checks.append(not is_irreducible(sq, p))
    checks.append(is_irreducible([1, 1, 0, 0, 1], p))  # x^4+x+1
    return float(sum(checks) / len(checks))


def bench_field_ext(seed: int = 0) -> dict[str, float]:
    return {"synthetic_field_ext": _bench_field_ext(seed)}
