"""Ben-Or/Tiwari sparse polynomial interpolation (SYNTHETIC bench)."""

from __future__ import annotations


def bm(seq: list[int], p: int) -> list[int]:
    """Berlekamp–Massey: minimal LFSR (connection poly) for seq over F_p."""
    c = [1]
    b = [1]
    l_, m_, bb = 0, 1, 1
    n_ = 0
    for n_ in range(len(seq)):
        d = seq[n_]
        for i in range(1, l_ + 1):
            d = (d + c[i] * seq[n_ - i]) % p
        if d == 0:
            m_ += 1
        elif 2 * l_ <= n_:
            t = list(c)
            coef = d * pow(bb, -1, p) % p
            while len(c) < len(b) + m_:
                c.append(0)
            for i in range(len(b)):
                c[i + m_] = (c[i + m_] - coef * b[i]) % p
            l_, b, bb, m_ = n_ + 1 - l_, t, d, 1
        else:
            coef = d * pow(bb, -1, p) % p
            while len(c) < len(b) + m_:
                c.append(0)
            for i in range(len(b)):
                c[i + m_] = (c[i + m_] - coef * b[i]) % p
            m_ += 1
    return c[: l_ + 1]


def roots_of(c: list[int], p: int) -> list[int]:
    """Roots of connection polynomial zeta(x)=c[0]+c[1]x+...+c[L]x^L in F_p."""
    return [x for x in range(p) if peval(c, x, p) == 0]


def peval(f: list[int], x: int, p: int) -> int:
    out = 0
    for cc in reversed(f):
        out = (out * x + cc) % p
    return out


def interpolate(vals: list[int], p: int, nvars_exp_base: int = 11) -> list[tuple[int, int]]:
    """Recover (exponent, coeff) terms of a t-sparse univariate f over F_p
    from evaluations v_i = f(alpha^i), alpha = nvars_exp_base."""
    t = (len(vals)) // 2
    c = bm(vals[: 2 * t], p)
    roots = roots_of(c, p)
    if len(roots) < len(c) - 1:
        return []
    # bm returns reciprocal locator: roots are m_j^{-1}
    exps = sorted(_discrete_log(pow(r, -1, p), nvars_exp_base, p) for r in roots)
    # coefficients: solve Vandermonde system over F_p

    coeffs: list[int] = []
    ms = [pow(nvars_exp_base, e, p) for e in exps]
    coeffs = _vandermonde(ms, vals, p)
    return sorted(zip(exps, coeffs, strict=True), key=lambda z: z[0])


def _discrete_log(x: int, base: int, p: int) -> int:
    v = 1
    for e in range(p):
        if v == x:
            return e
        v = v * base % p
    raise ValueError("no log")


def _vandermonde(ms: list[int], vals: list[int], p: int) -> list[int]:
    """Solve V c = vals where V[i][j] = ms[j]^i mod p."""
    n = len(ms)
    a = [[pow(ms[j], i, p) for j in range(n)] for i in range(n)]
    b = vals[:n]
    # Gaussian elimination over F_p
    for col in range(n):
        piv = next(r for r in range(col, n) if a[r][col] != 0)
        a[col], a[piv] = a[piv], a[col]
        b[col], b[piv] = b[piv], b[col]
        inv = pow(a[col][col], -1, p)
        a[col] = [x * inv % p for x in a[col]]
        b[col] = b[col] * inv % p
        for r in range(n):
            if r != col and a[r][col]:
                f = a[r][col]
                a[r] = [(a[r][k] - f * a[col][k]) % p for k in range(n)]
                b[r] = (b[r] - f * b[col]) % p
    return b


def _bench_sparse_interp(seed: int = 0) -> float:
    p = 103
    base = 5
    checks = []
    # f(x) = 3x^2 + 7x^5 mod 103; v_i = f(base^i)
    terms = {2: 3, 5: 7}
    vals = [sum(c * pow(base, e * i, p) for e, c in terms.items()) % p for i in range(6)]
    out = interpolate(vals, p, base)
    got = dict(out)
    checks.append(got == terms)
    # single term
    vals2 = [9 * pow(base, 3 * i, p) % p for i in range(4)]
    out2 = interpolate(vals2, p, base)
    checks.append(dict(out2) == {3: 9})
    # bm sanity: constant sequence -> C = [1]
    checks.append(bm([2, 2, 2, 2], p)[0] == 1)
    return sum(checks) / len(checks)


def bench_sparse_interp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sparse_interp": _bench_sparse_interp(seed)}
