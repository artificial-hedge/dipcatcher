"""Brace operations on Hochschild cochains (SYNTHETIC)."""

from __future__ import annotations


def cup(f: list[int], g: list[int]) -> list[int]:
    """Cup product of cochains on a 1-simplex algebra toy:
    (f cup g)(a, b, c) = f(a, b) * g(b, c) summed over split points."""
    n = len(f) + len(g)
    out = [0] * n
    for i, fv in enumerate(f):
        for j, gv in enumerate(g):
            out[i + j] += fv * gv
    return out


def prelie_bracket(f: list[int], g: list[int]) -> int:
    """f { g } - g { f } graded commutator toy: insertion count."""
    return sum(f) * len(g) - sum(g) * len(f)


def _bench_brace_operad(seed: int = 0) -> float:
    checks = []
    # cup product is associative
    a, b, c = [1, 0], [0, 1], [1, 1]
    checks.append(cup(cup(a, b), c) == cup(a, cup(b, c)))
    # cup length adds
    checks.append(len(cup(a, b)) == len(a) + len(b))
    # pre-Lie bracket satisfies graded Jacobi on degrees
    x = prelie_bracket([1], [1, 1])
    checks.append(x == 1 * 2 - 2 * 1)
    checks.append(x == 0)
    # brace of [2] with [1]: 2*1 - 1*1 = 1
    checks.append(prelie_bracket([2], [1]) == 1)
    return float(sum(checks) / len(checks))


def bench_brace_operad(seed: int = 0) -> dict[str, float]:
    return {"synthetic_brace_operad": _bench_brace_operad(seed)}
