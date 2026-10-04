"""Sumcheck protocol over F_p (SYNTHETIC bench)."""

from __future__ import annotations

import random

P = 104729


def eval_poly(coeffs: list[int], x: int, p: int = P) -> int:
    out = 0
    for c in reversed(coeffs):
        out = (out * x + c) % p
    return out


def total_sum(g, n: int, p: int = P) -> int:
    """Sum g(x1..xn) over all Boolean assignments."""
    import itertools

    return int(sum(g(bits) for bits in itertools.product([0, 1], repeat=n)) % p)


def round_poly(g, fixed: list[int], var: int, n: int, p: int = P) -> list[int]:
    """Univariate coeffs of sum over all vars after `var` (index 0-based).

    s_i(t) = sum_{b_{i+1}..b_{n-1}} g(fixed, t, b_rest); deg = deg of g in x_i.
    Coeffs recovered by evaluation at t=0..deg (small deg assumed <=4).
    """
    import itertools

    deg = 4  # generous cap for toy polys
    vals = []
    for t in range(deg + 1):
        s = 0
        rest = n - var - 1
        for bits in itertools.product([0, 1], repeat=rest):
            xs = fixed + [t] + list(bits)
            s = (s + g(xs)) % p
        vals.append(s)
    # interpolate degree<=deg poly through vals -> coeffs via Lagrange
    return _interp(vals, p)


def _interp(vals: list[int], p: int) -> list[int]:
    n = len(vals)
    out = [0] * n
    for i in range(n):
        num = [1]
        den = 1
        for j in range(n):
            if j == i:
                continue
            num = _pmul(num, [-j % p, 1], p)
            den = den * pow(i - j, -1, p) % p
        for k in range(n):
            out[k] = (out[k] + vals[i] * den * num[k]) % p
    return out


def _pmul(a: list[int], b: list[int], p: int) -> list[int]:
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % p
    return out


def run_sumcheck(g, n: int, claim: int, rng: random.Random, p: int = P) -> bool:
    """Verify claim H = sum g over Boolean cube; honest-prover protocol."""
    fixed: list[int] = []
    prev = claim
    for i in range(n):
        s_i = round_poly(g, fixed, i, n, p)
        if (eval_poly(s_i, 0, p) + eval_poly(s_i, 1, p)) % p != prev % p:
            return False
        r = rng.randrange(p)
        prev = eval_poly(s_i, r, p)
        fixed.append(r)
    return eval_poly([g(fixed)], 0, p) == prev


def bad_prover(g, n: int, claim: int, rng: random.Random, p: int = P) -> bool:
    """Prover that lies on the first round poly (constant +1 shift)."""
    fixed: list[int] = []
    prev = claim
    for i in range(n):
        s_i = round_poly(g, fixed, i, n, p)
        if i == 0:
            s_i = [(s_i[0] + 1) % p] + s_i[1:]
        if (eval_poly(s_i, 0, p) + eval_poly(s_i, 1, p)) % p != prev % p:
            return False
        r = rng.randrange(p)
        prev = eval_poly(s_i, r, p)
        fixed.append(r)
    return eval_poly([g(fixed)], 0, p) == prev


def _bench_sumcheck(seed: int = 0) -> float:
    rng = random.Random(20261231 + 1068)
    checks = []

    def g1(xs):
        return (xs[0] * xs[1] + xs[2]) % P

    checks.append(total_sum(g1, 3) == 6)  # x0x1 contributes 2, x2 contributes 4
    checks.append(run_sumcheck(g1, 3, 6, rng))
    checks.append(not run_sumcheck(g1, 3, 5, rng))
    checks.append(not bad_prover(g1, 3, 6, rng))

    def g2(xs):
        return (xs[0] + 2 * xs[1] * xs[1]) % P

    checks.append(run_sumcheck(g2, 2, total_sum(g2, 2), rng))
    return sum(checks) / len(checks)


def bench_sumcheck(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sumcheck": _bench_sumcheck(seed)}
