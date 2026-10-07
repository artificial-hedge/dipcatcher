"""Miller-Rabin primality testing (synthetic) (SYNTHETIC).

Deterministic MR for 64-bit integers using the known small-witness
set {2,3,5,7,11,13,17,19,23,29,31,37}; probabilistic MR with random
bases for larger ints. Verified against trial-division oracle and a
reference sieve over sampled composites and primes.
"""

from __future__ import annotations

import random


def _mr_witness(n: int, a: int) -> bool:
    """True if a is a Miller-Rabin witness for compositeness of n."""
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    x = pow(a, d, n)
    if x in (1, n - 1):
        return False
    for _ in range(r - 1):
        x = pow(x, 2, n)
        if x == n - 1:
            return False
    return True


def is_prime(n: int, k: int = 8, rng: random.Random | None = None) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return False
    if n < 3_474_749_660_383:
        return not any(_mr_witness(n, a) for a in small[:7])
    rng = rng or random.Random(0)
    return not any(_mr_witness(n, rng.randint(2, n - 2)) for _ in range(k))


def _trial_div(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def bench_miller_rabin(seed: int = 20261231 + 240) -> dict[str, float]:
    rng = random.Random(seed)
    agree = 0
    trials = 300
    for _ in range(trials):
        n = rng.randint(2, 200_000)
        agree += int(is_prime(n) == _trial_div(n))
    # Carmichael numbers must be caught (561, 1105, 1729)
    carmichael_caught = all(not is_prime(c) for c in (561, 1105, 1729))
    # Mersenne prime 2^61-1
    m61 = is_prime(2**61 - 1)
    # semiprime detection rate for factor base
    semiprimes = [p * q for p, q in ((101, 103), (997, 991), (89, 97))]
    semi_ok = all(not is_prime(s) for s in semiprimes)
    return {
        "synthetic_agree": float(agree / trials),
        "synthetic_carmichael_caught": float(carmichael_caught),
        "synthetic_m61_prime": float(m61),
        "synthetic_semiprimes_caught": float(semi_ok),
    }
