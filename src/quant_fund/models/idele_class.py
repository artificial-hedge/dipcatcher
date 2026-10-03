"""Product formula for Q (SYNTHETIC)."""

from __future__ import annotations


def v_p(x: int, p: int) -> int:
    n = 0
    x = abs(x)
    while x % p == 0 and x:
        x //= p
        n += 1
    return n


def product_formula(x_num: int, x_den: int) -> float:
    """prod_v |x|_v over all places: real abs times p-adic norms on
    numerator and denominator."""

    num, den = abs(x_num), abs(x_den)
    prod = abs(x_num / x_den)  # real place
    primes = set()
    for m in (num, den):
        d = 2
        mm = m
        while d * d <= mm:
            if mm % d == 0:
                primes.add(d)
                mm //= d
                d = 2
            else:
                d += 1
        if mm > 1:
            primes.add(mm)
    for p in primes:
        prod *= p ** (-(v_p(num, p) - v_p(den, p)))
    return prod


def _bench_idele_class(seed: int = 0) -> float:
    checks = []
    # product formula = 1 for any rational
    for a, b in [(3, 1), (7, 4), (12, 5), (2, 9), (1, 6)]:
        checks.append(abs(product_formula(a, b) - 1.0) < 1e-9)
    return float(sum(checks) / len(checks))


def bench_idele_class(seed: int = 0) -> dict[str, float]:
    return {"synthetic_idele_class": _bench_idele_class(seed)}
