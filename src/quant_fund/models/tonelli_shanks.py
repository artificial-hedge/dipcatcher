"""Tonelli-Shanks square roots modulo a prime (synthetic).

Computes r = sqrt(a) mod p for odd prime p via the Q/S decomposition
p-1 = Q·2^S, plus Euler's criterion for residuosity. Verified by
squaring back and cross-checking non-residues against pow().
"""

from __future__ import annotations

import random


def legendre(a: int, p: int) -> int:
    ls = pow(a % p, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls


def tonelli(a: int, p: int) -> int | None:
    a %= p
    if a == 0:
        return 0
    if p == 2:
        return a
    if legendre(a, p) != 1:
        return None
    if p % 4 == 3:
        return pow(a, (p + 1) // 4, p)
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while legendre(z, p) != -1:
        z += 1
    c = pow(z, q, p)
    x = pow(a, (q + 1) // 2, p)
    t = pow(a, q, p)
    m = s
    while t != 1:
        i, t2 = 1, t * t % p
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
            if i == m:
                return None
        b = pow(c, 1 << (m - i - 1), p)
        x = x * b % p
        t = t * b * b % p
        c = b * b % p
        m = i
    return x


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def bench_tonelli_shanks(seed: int = 20261231 + 242) -> dict[str, float]:
    rng = random.Random(seed)
    primes = [p for p in range(101, 5000) if _is_prime(p)]
    res_hits = res_trials = 0
    nr_hits = nr_trials = 0
    trials = 120
    for _ in range(trials):
        p = rng.choice(primes)
        a = rng.randint(0, p - 1)
        r = tonelli(a, p)
        if legendre(a, p) == 1 or a == 0:
            res_trials += 1
            res_hits += int(r is not None and r * r % p == a % p)
        else:
            nr_trials += 1
            nr_hits += int(r is None)
    # known case: sqrt(10) mod 13 = 6 or 7
    r = tonelli(10, 13)
    known = r is not None and r * r % 13 == 10
    return {
        "synthetic_agree": float(res_hits / max(1, res_trials)),
        "synthetic_known": float(known),
        "synthetic_nonres_agree": float(nr_hits / max(1, nr_trials)),
    }
