"""Pollard's rho integer factorization (synthetic).

Brent's improved cycle detection on f(x) = x² + c mod n with
batched gcd. Verified by recovering planted semiprimes and
cross-checking every returned factor is nontrivial and divides n.
"""

from __future__ import annotations

import random
from math import gcd


def pollard_rho(n: int, rng: random.Random, max_iter: int = 100_000) -> int:
    if n % 2 == 0:
        return 2
    y, c, m = rng.randint(1, n - 1), rng.randint(1, n - 1), 128
    g, r, q = 1, 1, 1
    xs = 0
    ys = 0
    it = 0
    while g == 1:
        xs = y
        for _ in range(r):
            y = (y * y % n + c) % n
        k = 0
        while k < r and g == 1:
            ys = y
            for _ in range(min(m, r - k)):
                y = (y * y % n + c) % n
                q = q * abs(xs - y) % n
                it += 1
            g = gcd(q, n)
            k += m
        r *= 2
        if it > max_iter:
            return n
    if g == n:
        while True:
            ys = (ys * ys % n + c) % n
            g = gcd(abs(xs - ys), n)
            if g > 1:
                break
    return g


def factor(n: int, rng: random.Random) -> list[int]:
    out: list[int] = []
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        d = pollard_rho(m, rng)
        if d == m:
            out.append(m)
        else:
            stack.extend([d, m // d])
    return sorted(out)


def _is_prime_trial(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def bench_pollard_rho(seed: int = 20261231 + 241) -> dict[str, float]:
    rng = random.Random(seed)
    # semiprimes: product of two primes ~1e3-1e5
    cases = [(1009, 1013), (7919, 7933), (31337, 61559)]
    found = 0
    for p, q in cases:
        n = p * q
        d = pollard_rho(n, rng)
        found += int(d in (p, q))
    # full factorization product check
    prod_ok = 0
    for p, q in cases:
        facs = factor(p * q, rng)
        prod = 1
        for f in facs:
            prod *= f
        prod_ok += int(prod == p * q and all(_is_prime_trial(f) for f in facs))
    # randomized small semiprimes
    agree = 0
    trials = 15
    primes = [p for p in range(50, 500) if _is_prime_trial(p)]
    for _ in range(trials):
        p, q = rng.choice(primes), rng.choice(primes)
        n = p * q
        # probabilistic: allow up to 4 restarts per instance
        d = n
        for _try in range(4):
            d = pollard_rho(n, rng)
            if d != n:
                break
        agree += int(d in (p, q) and n % d == 0)
    return {
        "synthetic_recovered": float(found / len(cases)),
        "synthetic_full_factor": float(prod_ok / len(cases)),
        "synthetic_agree": float(agree / trials),
    }
