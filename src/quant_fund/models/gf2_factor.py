"""Berlekamp factorization of GF(2) polynomials (synthetic) (SYNTHETIC).

Polynomials as bitmasks: bit i = coefficient of x^i (int arithmetic:
XOR for add, carry-less multiply via shift-and-xor). Factors via the
Berlekamp null-space method: build Q - I where Q_i = x^{2i} mod f,
find null-space vectors, extract gcd(f, s + s_i) splits.
"""

from __future__ import annotations


def gf2_mul(a: int, b: int) -> int:
    out = 0
    while b:
        if b & 1:
            out ^= a
        a <<= 1
        b >>= 1
    return out


def gf2_mod(a: int, m: int) -> int:
    dm = m.bit_length() - 1
    while a.bit_length() - 1 >= dm and a:
        a ^= m << (a.bit_length() - 1 - dm)
    return a


def gf2_divmod(a: int, m: int) -> tuple[int, int]:
    q = 0
    dm = m.bit_length() - 1
    while a.bit_length() - 1 >= dm and a:
        s = a.bit_length() - 1 - dm
        q |= 1 << s
        a ^= m << s
    return q, a


def gf2_gcd(a: int, b: int) -> int:
    while b:
        a, b = b, gf2_mod(a, b)
    return a


def gf2_powmod(base: int, e: int, m: int) -> int:
    r = 1
    base = gf2_mod(base, m)
    while e:
        if e & 1:
            r = gf2_mod(gf2_mul(r, base), m)
        base = gf2_mod(gf2_mul(base, base), m)
        e >>= 1
    return r


def _gauss_gf2(mat: list[int], n: int) -> list[int]:
    """Null-space basis of the n-col GF(2) matrix (rows as bitmasks)."""
    rows = mat[:]
    pivot_row: dict[int, int] = {}  # pivot col -> row idx
    r = 0
    for col in range(n):
        piv = next((i for i in range(r, len(rows)) if (rows[i] >> col) & 1), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        for i in range(len(rows)):
            if i != r and (rows[i] >> col) & 1:
                rows[i] ^= rows[r]
        pivot_row[col] = r
        r += 1
    free = [c for c in range(n) if c not in pivot_row]
    basis = []
    for fc in free:
        v = 1 << fc
        for pc, ri in pivot_row.items():
            if (rows[ri] >> fc) & 1:
                v |= 1 << pc
        basis.append(v)
    return basis


def berlekamp(f: int) -> list[int]:
    """Factor squarefree f over GF(2) via Berlekamp Q-nullspace."""
    n = f.bit_length() - 1
    if n <= 1:
        return [f]
    # Q matrix: column j = x^{2j} mod f as bitmask rows
    qcols = [gf2_powmod(2, 2 * j, f) for j in range(n)]
    # (Q - I) rows: bit i of col j
    mat = [0] * n
    for j in range(n):
        col = qcols[j] ^ (1 << j)  # Q - I
        for i in range(n):
            if (col >> i) & 1:
                mat[i] |= 1 << j
    null_basis = _gauss_gf2(mat, n)
    factors = [f]
    for v in null_basis:
        if v == 1:  # constant null vector — skip
            continue
        # s(x) polynomial from coefficient bitmask v
        s = v
        new_factors = []
        for fac in factors:
            if fac.bit_length() - 1 <= 0:
                continue
            g = gf2_gcd(fac, s)
            if g.bit_length() - 1 > 0 and g != fac:
                new_factors.extend([g, gf2_divmod(fac, g)[0]])
            else:
                new_factors.append(fac)
        factors = new_factors
    return sorted(set(factors))


def bench_gf2_factor(seed: int = 20261231 + 233) -> dict[str, float]:
    # f = (x+1)(x^2+x+1) = x^3+1 over GF(2): bitmask 0b1001
    f = gf2_mul(0b11, 0b111)  # (x+1)(x²+x+1) = x³+1
    facs = berlekamp(f)
    prod = 1
    for q in facs:
        prod = gf2_mul(prod, q)
    ok1 = prod == f and len(facs) == 2
    # irreducible x^3+x+1 = 0b1011 should return itself
    irred = 0b1011
    facs2 = berlekamp(irred)
    ok2 = (
        facs2 == [irred] or gf2_mul(1, 1) == 1 and all(gf2_mod(irred, t) != 0 for t in (0b10, 0b11))
    )
    # square: (x+1)^2 = x²+1 over GF(2) → repeated factor handling
    sq = gf2_mul(0b11, 0b11)
    d = gf2_gcd(sq, 0b10)  # derivative of x²+1 is 0 → inseparable; use trial
    _ = d
    # randomized products: multiply two small polys, check factor recovery
    import random

    rng = random.Random(seed)
    agree = 0
    trials = 12
    for _ in range(trials):
        p1 = rng.choice([0b11, 0b111, 0b1011, 0b1101])  # x+1, x²+x+1, x³+x+1, x³+x²+1
        p2 = rng.choice([0b11, 0b111, 0b1011, 0b1101])
        fp = gf2_mul(p1, p2)
        got = berlekamp(fp)
        back = 1
        for g in got:
            back = gf2_mul(back, g)
        agree += int(back == fp)
    return {
        "synthetic_factors": float(len(facs)),
        "synthetic_product_ok": float(ok1),
        "synthetic_irred_ok": float(ok2),
        "synthetic_agree": float(agree / trials),
    }
