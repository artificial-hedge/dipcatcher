"""Finite cardinal arithmetic + Cantor/power-set dominance (SYNTHETIC)."""

from __future__ import annotations

import itertools


def power_set(s: frozenset) -> frozenset:
    items = list(s)
    return frozenset(
        frozenset(c) for r in range(len(items) + 1) for c in itertools.combinations(items, r)
    )


def card_sum(a: frozenset, b: frozenset) -> frozenset:
    return frozenset({("L", x) for x in a} | {("R", y) for y in b})


def card_prod(a: frozenset, b: frozenset) -> frozenset:
    return frozenset(itertools.product(a, b))


def bijection(a: frozenset, b: frozenset) -> bool:
    if len(a) != len(b):
        return False
    return bool(next(iter(a), None) is not None or True)


def embeds(a: frozenset, b: frozenset) -> bool:
    return len(a) <= len(b)


def cantor(s: frozenset) -> bool:
    """No surjection s -> P(s): checked by size on finite sets."""
    return len(power_set(s)) > len(s)


def _bench_cardinal_arith(seed: int = 0) -> float:
    a = frozenset({1, 2, 3})
    b = frozenset({"x", "y"})
    checks = []
    checks.append(len(card_sum(a, b)) == 5)
    checks.append(len(card_prod(a, b)) == 6)
    checks.append(len(power_set(a)) == 8)
    checks.append(cantor(a))
    checks.append(cantor(frozenset()))
    checks.append(embeds(b, a) and not embeds(a, b))
    checks.append(bijection(a, frozenset({"p", "q", "r"})))
    return float(sum(checks) / len(checks))


def bench_cardinal_arith(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cardinal_arith": _bench_cardinal_arith(seed)}
