"""Splitting of primes in Q(sqrt(d)) via Kronecker symbol (SYNTHETIC)."""

from __future__ import annotations


def legendre(a: int, p: int) -> int:
    """Legendre symbol (a/p) = a^((p-1)/2) mod p in {-1,0,1}."""
    if a % p == 0:
        return 0
    r = pow(a, (p - 1) // 2, p)
    return -1 if r == p - 1 else r


def split_type(d: int, p: int) -> str:
    """p splits iff (d/p) = 1, inert iff -1, ramified iff 0."""
    s = legendre(d, p)
    if s == 0:
        return "ramified"
    return "split" if s == 1 else "inert"


def _bench_ramification(seed: int = 0) -> float:
    checks = []
    # 2 splits in Q(sqrt(7))? (7/3)... use p odd, d = 5: (5/11) = 1 split
    checks.append(split_type(5, 11) == "split")
    # (5/7) = -1: inert
    checks.append(split_type(5, 7) == "inert")
    # 5 ramifies in Q(sqrt(5))
    checks.append(split_type(5, 5) == "ramified")
    # (2/7) = 1: 7 splits in Q(sqrt(2))
    checks.append(split_type(2, 7) == "split")
    # (2/5) = -1: inert in Q(sqrt(2)) at 5
    checks.append(split_type(2, 5) == "inert")
    # legendre values
    checks.append(legendre(3, 7) == -1)
    checks.append(legendre(2, 7) == 1)
    return float(sum(checks) / len(checks))


def bench_ramification(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ramification": _bench_ramification(seed)}
