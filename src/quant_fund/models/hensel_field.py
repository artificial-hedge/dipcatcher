"""Hensel lifting: p-adic Newton iteration (SYNTHETIC)."""

from __future__ import annotations


def lift_root(a: int, p: int, power: int, f, fprime) -> int | None:
    """Lift a simple root of f mod p to a root mod p^power via
    x' = x - f(x)/f'(x) mod next modulus."""
    mod = p
    x = a % p
    while mod < power:
        nxt = mod * p
        fx = f(x) % nxt
        # f'(x) is a unit mod p for a simple root
        fp = fprime(x) % p
        if fp == 0:
            return None
        inv = pow(fp, -1, p)
        x = (x - fx * inv) % nxt
        mod = nxt
    return x


def _bench_hensel_field(seed: int = 0) -> float:
    checks = []
    # sqrt(2) mod 7: x=3 (9 = 2 mod 7); lift to mod 49
    def f(t: int) -> int:
        return t * t - 2

    def fp(t: int) -> int:
        return 2 * t

    r = lift_root(3, 7, 49, f, fp)
    checks.append(r is not None and r * r % 49 == 2)
    # other root lifts from 4
    r2 = lift_root(4, 7, 49, f, fp)
    checks.append(r2 is not None and r2 * r2 % 49 == 2)
    # the two lifts are distinct mod 49
    checks.append(r is not None and r2 is not None and r != r2)
    # roots sum to 0 mod 49 (x^2-2 -> roots are negatives)
    checks.append(r is not None and r2 is not None and (r + r2) % 49 == 0)
    # x^3 - 2 = 0 mod 5: x=3 (27=2 mod5); lift to 25
    r3 = lift_root(3, 5, 25, lambda t: t**3 - 2, lambda t: 3 * t * t)
    checks.append(r3 is not None and r3**3 % 25 == 2)
    return float(sum(checks) / len(checks))


def bench_hensel_field(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hensel_field": _bench_hensel_field(seed)}
