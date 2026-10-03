"""Spec of finite commutative rings: prime ideals, maximal ideals (SYNTHETIC)."""

from __future__ import annotations

from quant_fund.models.ring_ideals import ideal_in_zn, is_ideal_zn


def is_prime_ideal_zn(n: int, s: frozenset[int]) -> bool:
    """P prime iff ab in P => a in P or b in P (and P != whole ring)."""
    if not is_ideal_zn(n, s) or len(s) == n:
        return False
    for a in range(n):
        for b in range(n):
            if (a * b) % n in s and a not in s and b not in s:
                return False
    return True


def is_maximal_zn(n: int, s: frozenset[int]) -> bool:
    """Maximal proper ideal."""
    if not is_prime_ideal_zn(n, s):
        return False
    for a in range(n):
        bigger = ideal_in_zn(n, s | {a})
        if bigger != s and len(bigger) != n:
            return False
    return True


def spec_zn(n: int) -> frozenset[frozenset[int]]:
    from itertools import combinations

    out = set()
    elems = list(range(n))
    for r in range(1, n):
        for c in combinations(elems, r):
            s = frozenset(c)
            if is_prime_ideal_zn(n, s):
                out.add(s)
    return frozenset(out)


def _bench_spec_ring(seed: int = 0) -> float:
    checks = []
    checks.append(is_prime_ideal_zn(6, frozenset({0, 2, 4})))  # mod 2 ideal
    checks.append(is_prime_ideal_zn(6, frozenset({0, 3})))  # mod 3 ideal
    checks.append(not is_prime_ideal_zn(6, frozenset({0})))  # zero ideal not prime (Z6 not domain)
    sp = spec_zn(6)
    checks.append(len(sp) == 2)
    checks.append(is_maximal_zn(6, frozenset({0, 2, 4})))
    # Spec(Z/4) = { (2) } single point
    checks.append(len(spec_zn(4)) == 1)
    return float(sum(checks) / len(checks))


def bench_spec_ring(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spec_ring": _bench_spec_ring(seed)}
