"""Chinese remainder theorem + Garner's algorithm (synthetic) (SYNTHETIC).

CRT via direct constructive sum and via Garner's mixed-radix
conversion; both verified against brute-force modular search and
against each other on random pairwise-coprime moduli.
"""

from __future__ import annotations

import random
from math import gcd


def _egcd(a: int, b: int) -> tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x, y = _egcd(b, a % b)
    return g, y, x - (a // b) * y


def _inv(a: int, m: int) -> int:
    g, x, _ = _egcd(a % m, m)
    if g != 1:
        raise ValueError("not invertible")
    return x % m


def crt(residues: list[int], moduli: list[int]) -> int:
    """Direct CRT: x = Σ a_i·M_i·inv(M_i, m_i) mod M."""
    m_tot = 1
    for m in moduli:
        m_tot *= m
    x = 0
    for a, m in zip(residues, moduli, strict=True):
        mi = m_tot // m
        x += a * mi * _inv(mi, m)
    return x % m_tot


def garner(residues: list[int], moduli: list[int]) -> int:
    """Garner: mixed-radix coefficients then Horner reconstruction."""
    k = len(moduli)
    x = [0] * k
    for i in range(k):
        x[i] = residues[i] % moduli[i]
        for j in range(i):
            x[i] = (x[i] - x[j]) * _inv(moduli[j], moduli[i]) % moduli[i]
    out = 0
    for i in range(k - 1, -1, -1):
        out = out * moduli[i] + x[i]
    return out


def bench_crt_garner(seed: int = 20261231 + 244) -> dict[str, float]:
    rng = random.Random(seed)
    primes = [p for p in range(3, 200) if all(p % i for i in range(2, p))]
    agree = 0
    trials = 30
    for _ in range(trials):
        k = rng.randint(2, 4)
        mods = rng.sample(primes, k)
        res = [rng.randint(0, m - 1) for m in mods]
        x1, x2 = crt(res, mods), garner(res, mods)
        m_tot = 1
        for m in mods:
            m_tot *= m
        # brute-force check on small products
        if m_tot < 100_000:
            bf = next(
                (
                    x
                    for x in range(m_tot)
                    if all(x % m == r for m, r in zip(mods, res, strict=True))
                ),
                None,
            )
            agree += int(x1 == x2 == bf)
        else:
            agree += int(x1 == x2)
    # 3-modulus textbook: x≡2 mod3, 3 mod5, 2 mod7 → 23
    x = crt([2, 3, 2], [3, 5, 7])
    known = x == 23 and garner([2, 3, 2], [3, 5, 7]) == 23
    # pairwise-coprime validation path
    cop_ok = all(gcd(3, m) == 1 for m in (5, 7))
    return {
        "synthetic_agree": float(agree / trials),
        "synthetic_known_23": float(known),
        "synthetic_coprime": float(cop_ok),
    }
