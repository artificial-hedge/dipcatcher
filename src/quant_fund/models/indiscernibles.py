"""Order indiscernibles: same-order tuples satisfy same types (SYNTHETIC)."""

from __future__ import annotations

from fractions import Fraction


def order_type(t: tuple[Fraction, ...]) -> tuple[int, ...]:
    """The order type of a tuple: ranks of sorted positions."""
    srt = sorted(t)
    return tuple(srt.index(x) for x in t)


def same_atomic_type(t1: tuple[Fraction, ...], t2: tuple[Fraction, ...]) -> bool:
    """In pure DLO, tuples have the same atomic type iff they have the
    same order type (and equality pattern)."""
    if len(t1) != len(t2):
        return False
    eq1 = {i: j for i, a in enumerate(t1) for j, b in enumerate(t1) if a == b}
    eq2 = {i: j for i, a in enumerate(t2) for j, b in enumerate(t2) if a == b}
    return order_type(t1) == order_type(t2) and eq1 == eq2


def _bench_indiscernibles(seed: int = 0) -> float:
    checks = []
    s = [Fraction(i) for i in range(10)]
    # increasing 2-tuples all share the same type
    t1 = (s[0], s[5])
    t2 = (s[3], s[7])
    checks.append(same_atomic_type(t1, t2))
    # reversed order differs
    checks.append(not same_atomic_type((s[1], s[0]), t1))
    # triples with equal first two coordinates differ from distinct ones
    checks.append(not same_atomic_type((s[0], s[0], s[2]), (s[0], s[1], s[2])))
    # all increasing triples equivalent — indiscernibility
    checks.append(
        all(
            same_atomic_type((s[i], s[j], s[k]), (s[0], s[1], s[2]))
            for i in range(10)
            for j in range(i + 1, 10)
            for k in range(j + 1, 10)
        )
    )
    # order_type of sorted tuple is identity
    checks.append(order_type((s[1], s[3], s[8])) == (0, 1, 2))
    checks.append(order_type((s[8], s[1], s[3])) == (2, 0, 1))
    return float(sum(checks) / len(checks))


def bench_indiscernibles(seed: int = 0) -> dict[str, float]:
    return {"synthetic_indiscernibles": _bench_indiscernibles(seed)}
