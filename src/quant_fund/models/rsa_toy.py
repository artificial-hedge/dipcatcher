"""SYNTHETIC textbook RSA (small primes — correctness demo only).

Keygen, encrypt/decrypt, sign/verify, and key-transport identity
Enc(pk, Dec(sk, m)) == m verified on random messages. NOT secure padding.
"""

from __future__ import annotations

import math
import random


def _isprime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def keygen(rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    while True:
        p = rng.randrange(200, 400) | 1
        if _isprime(p):
            break
    while True:
        q = rng.randrange(400, 600) | 1
        if _isprime(q) and q != p:
            break
    n, phi = p * q, (p - 1) * (q - 1)
    e = 65537
    while math.gcd(e, phi) != 1:
        e = rng.randrange(3, phi) | 1
    d = pow(e, -1, phi)
    return (n, e), (n, d)


def bench_rsa_toy(seed: int = 20261231 + 420) -> dict[str, float]:
    rng = random.Random(seed)
    rt = sig = tamper = 0
    trials = 25
    for _ in range(trials):
        pub, priv = keygen(rng)
        n, e = pub
        m = rng.randrange(2, n - 1)
        c = pow(m, e, n)
        rt += int(pow(c, priv[1], n) == m)
        # sign = hash^m... textbook: s = m^d, verify s^e == m
        s = pow(m, priv[1], n)
        sig += int(pow(s, e, n) == m)
        # tampered ciphertext fails
        c2 = (c + 1) % n
        tamper += int(pow(c2, priv[1], n) != m)
    return {
        "synthetic_roundtrip": float(rt / trials),
        "synthetic_signature_verifies": float(sig / trials),
        "synthetic_tamper_detected": float(tamper / trials),
    }
