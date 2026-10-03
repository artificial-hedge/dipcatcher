"""Chinese remainder for polynomials and integers (SYNTHETIC bench)."""

from __future__ import annotations


def crt_int(rems: list[int], mods: list[int]) -> int:
    """x ≡ rems[i] mod mods[i], pairwise coprime mods."""
    m = 1
    for mi in mods:
        m *= mi
    x = 0
    for r, mi in zip(rems, mods, strict=True):
        mi_hat = m // mi
        x += r * mi_hat * pow(mi_hat, -1, mi)
    return x % m


def poly_eval(f: list[int], x: int, p: int) -> int:
    out = 0
    for c in reversed(f):
        out = (out * x + c) % p
    return out


def poly_interp_mod(xs: list[int], ys: list[int], p: int) -> list[int]:
    """Lagrange interpolation mod p."""
    n = len(xs)
    out = [0]
    for i in range(n):
        num = [1]
        den = 1
        for j in range(n):
            if i == j:
                continue
            num = _pmul_mod(num, [-xs[j] % p, 1], p)
            den = den * pow(xs[i] - xs[j], -1, p) % p
        term = [c * den * ys[i] % p for c in num]
        out = _padd_mod(out, term, p)
    return out


def _pmul_mod(a: list[int], b: list[int], p: int) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % p
    while out and out[-1] == 0:
        out.pop()
    return out or [0]


def _padd_mod(a: list[int], b: list[int], p: int) -> list[int]:
    n = max(len(a), len(b))
    return [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(n)] and [
        ((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)) % p for i in range(n)
    ]


def poly_crt(r1: list[int], m1: list[int], r2: list[int], m2: list[int], p: int) -> list[int]:
    """f ≡ r1 mod m1, f ≡ r2 mod m2 (coprime) -> f mod m1*m2 via CRT on coefficients."""
    # find s,t with s*m1 + t*m2 = 1 -> f = r1*t*m2 + r2*s*m1
    a, b = list(m1), list(m2)
    u1, u2 = [1], [0]
    v1, v2 = [0], [1]
    while b:
        q, r = _pdivmod(a, b, p)
        a, b = b, r
        u1, u2 = u2, _psub(u1, _pmul_mod(q, u2, p), p)
        v1, v2 = v2, _psub(v1, _pmul_mod(q, v2, p), p)
    s, t = u1, v1  # u1*m1 + v1*m2 = 1
    f = _padd_mod(_pmul_mod(_pmul_mod(r1, t, p), m2, p), _pmul_mod(_pmul_mod(r2, s, p), m1, p), p)
    _, f = _pdivmod(f, _pmul_mod(m1, m2, p), p)
    return [c % p for c in f]


def _pdivmod(a: list[int], b: list[int], p: int) -> tuple[list[int], list[int]]:
    a = list(a)
    q = [0] * max(1, len(a))
    while len(a) >= len(b) and a:
        k = len(a) - len(b)
        c = a[-1] * pow(b[-1], -1, p) % p
        q[k] = c
        for i in range(len(b)):
            a[i + k] = (a[i + k] - c * b[i]) % p
        while a and a[-1] == 0:
            a.pop()
    return [x % p for x in q], a


def _psub(a: list[int], b: list[int], p: int) -> list[int]:
    n = max(len(a), len(b))
    out = [((a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)) % p for i in range(n)]
    while out and out[-1] == 0:
        out.pop()
    return out or [0]


def _bench_poly_crt(seed: int = 0) -> float:
    checks = []
    checks.append(crt_int([2, 3], [5, 7]) == 17)
    checks.append(crt_int([1, 2, 3], [3, 5, 7]) == 52)
    p = 7
    f = poly_interp_mod([0, 1, 2], [1, 3, 2], p)
    checks.append(all(poly_eval(f, x, p) == y for x, y in zip([0, 1, 2], [1, 3, 2], strict=True)))
    # f = 2x+1 mod 7: f mod (x-0)=1, f mod (x-1)=3 -> crt combine to f mod x(x-1)
    g = poly_crt([1], [0, 1], [3], [6, 1], p)
    checks.append(poly_eval(g, 0, p) == 1 and poly_eval(g, 1, p) == 3)
    return sum(checks) / len(checks)


def bench_poly_crt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_poly_crt": _bench_poly_crt(seed)}
