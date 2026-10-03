"""Divisors on a curve: deg(principal divisor) = 0 (SYNTHETIC)."""

from __future__ import annotations

Div = dict[int, int]  # point index -> order


def degree(d: Div) -> int:
    return sum(d.values())


def div_of_function(zeros: dict[int, int], poles: dict[int, int]) -> Div:
    out = dict(zeros)
    for p, v in poles.items():
        out[p] = out.get(p, 0) - v
    return out


def add(d1: Div, d2: Div) -> Div:
    out = dict(d1)
    for p, v in d2.items():
        out[p] = out.get(p, 0) + v
        if out[p] == 0:
            del out[p]
    return out


def is_effective(d: Div) -> bool:
    return all(v > 0 for v in d.values())


def _bench_divisor_group(seed: int = 0) -> float:
    checks = []
    # principal divisor: zeros - poles has degree 0
    f = div_of_function({0: 2, 1: 1}, {5: 3})
    checks.append(degree(f) == 0)
    checks.append(f == {0: 2, 1: 1, 5: -3})
    # group law
    d = {0: 1, 2: 3}
    e = {2: -3, 4: 1}
    checks.append(add(d, e) == {0: 1, 4: 1})
    checks.append(degree(add(d, e)) == degree(d) + degree(e))
    # effectivity
    checks.append(is_effective({0: 2}))
    checks.append(not is_effective({0: 2, 1: -1}))
    # principal divisors form subgroup: sum of principals is principal
    checks.append(degree(add(f, f)) == 0)
    return float(sum(checks) / len(checks))


def bench_divisor_group(seed: int = 0) -> dict[str, float]:
    return {"synthetic_divisor_group": _bench_divisor_group(seed)}
