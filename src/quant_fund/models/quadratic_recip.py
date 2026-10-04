"""Quadratic reciprocity: Legendre/Jacobi symbols, Euler criterion (SYNTHETIC)."""

from __future__ import annotations


def legendre(a: int, p: int) -> int:
    """(a|p) for odd prime p via Euler criterion."""
    a %= p
    if a == 0:
        return 0
    return 1 if pow(a, (p - 1) // 2, p) == 1 else -1


def is_qr(a: int, p: int) -> bool:
    return any(x * x % p == a % p for x in range(p))


def jacobi(a: int, n: int) -> int:
    """Jacobi symbol (a|n) for odd n via factorization."""
    out = 1
    m = n
    p = 3
    factors = []
    while p * p <= m:
        while m % p == 0:
            factors.append(p)
            m //= p
        p += 2
    if m > 1:
        factors.append(m)
    if n % 2 == 0:
        return 0
    for f in factors:
        out *= legendre(a, f)
    return out


def _bench_quadratic_recip(seed: int = 0) -> float:
    checks = []
    checks.append(legendre(2, 7) == 1)
    checks.append(legendre(3, 7) == -1)
    checks.append(legendre(0, 7) == 0)
    # brute-force agreement
    checks.append(
        all(
            legendre(a, 11) == (0 if a % 11 == 0 else (1 if is_qr(a, 11) else -1))
            for a in range(22)
        )
    )
    # quadratic reciprocity on (3,5): (3|5)(5|3) = -1 * -1 = 1 = (-1)^{((3-1)/2)((5-1)/2)}=1
    lhs = legendre(3, 5) * legendre(5, 3)
    rhs = (-1) ** (((3 - 1) // 2) * ((5 - 1) // 2))
    checks.append(lhs == rhs)
    checks.append(jacobi(2, 15) == legendre(2, 3) * legendre(2, 5))
    return float(sum(checks) / len(checks))


def bench_quadratic_recip(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quadratic_recip": _bench_quadratic_recip(seed)}
