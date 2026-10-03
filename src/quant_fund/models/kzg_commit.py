"""KZG polynomial commitment over a toy bilinear group.

Group elements live in additive Z_n notation (scalars); the pairing is
e(a, b) = a*b mod n — a genuine bilinear map for this cyclic toy group.
SRS: powers of a hidden tau (tau^i in the exponent). Commit C = p(tau)*1;
open at z reveals v=p(z) and proof pi = q(tau) where q(x)=(p(x)-v)/(x-z).
Verifier checks e(pi, tau-z) == e(C-v, 1) i.e. pi*(tau-z) == C - v.
"""

from __future__ import annotations

import random as _r

_SEED = 20261231 + 1049

N = 104729  # prime group order (toy)


def srs(tau: int, degree: int, n: int = N) -> list[int]:
    """Powers of tau in the exponent: [1, tau, tau^2, ...]."""
    out = [1]
    for _ in range(degree):
        out.append(out[-1] * tau % n)
    return out


def _eval(c: list[int], x: int, n: int = N) -> int:
    acc = 0
    for a in reversed(c):
        acc = (acc * x + a) % n
    return acc


def commit(coeffs: list[int], powers: list[int], n: int = N) -> int:
    if len(coeffs) > len(powers):
        raise ValueError("SRS too small for polynomial degree")
    return sum(c * powers[i] for i, c in enumerate(coeffs)) % n


def _div_linear(num: list[int], z: int, n: int = N) -> list[int]:
    """Divide num by (x - z): returns quotient coeffs (rem is num(z))."""
    q = [0] * (len(num) - 1)
    acc = 0
    for i in range(len(num) - 1, 0, -1):
        acc = (num[i] + acc * z) % n
        q[i - 1] = acc
    return q


def open_at(coeffs: list[int], z: int, powers: list[int], n: int = N) -> tuple[int, int]:
    """Returns (v, pi): value and commitment to quotient poly."""
    v = _eval(coeffs, z, n)
    q = _div_linear(coeffs, z, n)
    pi = sum(c * powers[i] for i, c in enumerate(q)) % n
    return v, pi


def verify(c: int, z: int, v: int, pi: int, tau: int, n: int = N) -> bool:
    """e(pi, tau - z) == e(c - v, 1): pi*(tau-z) == c - v (mod n)."""
    return (pi * (tau - z)) % n == (c - v) % n


def bench_kzg_commit(seed: int = _SEED) -> dict[str, float]:
    rng = _r.Random(seed)
    checks: list[bool] = []
    tau = rng.randrange(2, 1000)
    powers = srs(tau, 8)
    # p(x) = 3 + 5x + 7x^2
    p = [3, 5, 7]
    c = commit(p, powers)
    z = 4
    v, pi = open_at(p, z, powers)
    checks.append(v == _eval(p, z) and v == (3 + 5 * 4 + 7 * 16) % N)
    checks.append(verify(c, z, v, pi, tau))
    # wrong value rejected
    checks.append(not verify(c, z, (v + 1) % N, pi, tau))
    # wrong proof rejected
    checks.append(not verify(c, z, v, (pi + 1) % N, tau))
    # batch: second poly degree-3 at another point
    p2 = [11, 0, 2, 9]
    c2 = commit(p2, powers)
    v2, pi2 = open_at(p2, 7, powers)
    checks.append(verify(c2, 7, v2, pi2, tau) and v2 == _eval(p2, 7))
    # opening at a different point's proof fails
    checks.append(not verify(c2, 5, v2, pi2, tau))
    return {"synthetic_kzg_commit": float(sum(checks)) / len(checks)}
