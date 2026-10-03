"""Legendre / Jacobi symbols and reciprocity identities (SYNTHETIC)."""

from __future__ import annotations


def legendre(a: int, p: int) -> int:
    """(a/p) for odd prime p."""
    if a % p == 0:
        return 0
    r = pow(a, (p - 1) // 2, p)
    return -1 if r == p - 1 else r


def minus_one_symbol(p: int) -> int:
    """(-1/p) = (-1)^((p-1)/2)."""
    return 1 if (p - 1) // 2 % 2 == 0 else -1


def two_symbol(p: int) -> int:
    """(2/p) = (-1)^((p^2-1)/8)."""
    return 1 if p % 8 in (1, 7) else -1


def _bench_artin_symbol(seed: int = 0) -> float:
    checks = []
    # (5/11) = 1 since 4^2 = 5 mod 11
    checks.append(legendre(5, 11) == 1)
    # (3/7) = -1
    checks.append(legendre(3, 7) == -1)
    # multiplicative: (6/7) = (2/7)(3/7) = -1
    checks.append(legendre(6, 7) == legendre(2, 7) * legendre(3, 7))
    # (-1/5) = 1 (5 == 1 mod 4), (-1/7) = -1
    checks.append(minus_one_symbol(5) == 1)
    checks.append(minus_one_symbol(7) == -1)
    # (2/p) checks: +1 at 7,17; -1 at 3,5
    checks.append(two_symbol(7) == 1 and two_symbol(17) == 1)
    checks.append(two_symbol(3) == -1 and two_symbol(5) == -1)
    # legendre agrees with two_symbol at a=2
    checks.append(legendre(2, 7) == two_symbol(7))
    return float(sum(checks) / len(checks))


def bench_artin_symbol(seed: int = 0) -> dict[str, float]:
    return {"synthetic_artin_symbol": _bench_artin_symbol(seed)}
