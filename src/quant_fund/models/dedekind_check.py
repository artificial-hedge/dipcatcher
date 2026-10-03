"""Z[sqrt(-5)] fails unique factorization: 6 = 2*3 = (1+sqrt-5)(1-sqrt-5) (SYNTHETIC)."""

from __future__ import annotations

# elements as (a, b) for a + b*sqrt(-5)


def norm(e: tuple[int, int]) -> int:
    return e[0] * e[0] + 5 * e[1] * e[1]


def mul(x: tuple[int, int], y: tuple[int, int]) -> tuple[int, int]:
    return (x[0] * y[0] - 5 * x[1] * y[1], x[0] * y[1] + x[1] * y[0])


def is_irreducible_z5(e: tuple[int, int]) -> bool:
    """Check irreducibility by norm brute force."""
    n = norm(e)
    if n <= 1:
        return False
    for a in range(-10, 11):
        for b in range(-10, 11):
            c = _conj_quot(e, (a, b))
            if c is not None and 1 < norm(c) < n:
                return False
    return True


def _conj_quot(e: tuple[int, int], d: tuple[int, int]) -> tuple[int, int] | None:
    """e/d in Z[sqrt-5] if it exists (exact division)."""
    nd = norm(d)
    a = e[0] * d[0] + 5 * e[1] * d[1]
    b = e[1] * d[0] - e[0] * d[1]
    if a % nd or b % nd:
        return None
    return (a // nd, b // nd)


def _bench_dedekind_check(seed: int = 0) -> float:
    checks = []
    two = (2, 0)
    three = (3, 0)
    p1 = (1, 1)
    p2 = (1, -1)
    # both factorizations of 6
    checks.append(norm(mul(two, three)) == 36)
    checks.append(norm(mul(p1, p2)) == 36)
    checks.append(mul(two, three) == (6, 0))
    checks.append(mul(p1, p2) == (6, 0))
    # 2 does not divide (1+sqrt-5): quotient non-integral
    checks.append(_conj_quot(p1, two) is None)
    # norms are multiplicative
    checks.append(norm(mul(p1, three)) == norm(p1) * norm(three))
    return float(sum(checks) / len(checks))


def bench_dedekind_check(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dedekind_check": _bench_dedekind_check(seed)}
