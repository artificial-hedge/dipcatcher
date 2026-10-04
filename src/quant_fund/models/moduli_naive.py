"""Counting elliptic curves over GF(p) by j-invariant (SYNTHETIC)."""

from __future__ import annotations


def j_invariant(a: int, b: int, p: int) -> int:
    """j = 1728 * 4a^3 / (4a^3 + 27b^2) mod p."""
    num = (4 * a**3) % p
    den = (num + 27 * b * b) % p
    if den == 0:
        return -1  # singular, not an elliptic curve
    return (1728 * num * pow(den, -1, p)) % p


def is_singular(a: int, b: int, p: int) -> bool:
    return (4 * a**3 + 27 * b * b) % p == 0


def count_js(p: int) -> set[int]:
    return {j_invariant(a, b, p) for a in range(p) for b in range(p) if not is_singular(a, b, p)}


def _bench_moduli_naive(seed: int = 0) -> float:
    checks = []
    p = 7
    js = count_js(p)
    # every j in GF(7) is achieved by some curve
    checks.append(js == set(range(7)))
    # singular cubics have zero discriminant
    checks.append(is_singular(0, 0, p))
    checks.append(not is_singular(1, 1, p))
    # j is an invariant: same curve scaled u^4 a, u^6 b gives same j
    u = 2
    checks.append(j_invariant(1, 1, p) == j_invariant((u**4) % p, (u**6) % p, p))
    # j=0 curve y^2 = x^3 + 1 exists
    checks.append(j_invariant(0, 1, p) == 0)
    return float(sum(checks) / len(checks))


def bench_moduli_naive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moduli_naive": _bench_moduli_naive(seed)}
