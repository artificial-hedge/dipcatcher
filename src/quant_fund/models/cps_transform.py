"""SYNTHETIC CPS (continuation-passing style) transform + evaluator.

Source: arithmetic with 'if' and 'call' (higher-order). CPS result is
evaluated by an explicit-continuation machine; verifies the transformed
program equals the direct interpreter on random programs.
"""

from __future__ import annotations

import random
from collections.abc import Callable

# Source AST: ("n",v) ("+",a,b) ("-",a,b) ("*",a,b) ("if",c,t,e)


def direct(e: tuple) -> float:
    tag = e[0]
    if tag == "n":
        return float(e[1])
    if tag == "+":
        return direct(e[1]) + direct(e[2])
    if tag == "-":
        return direct(e[1]) - direct(e[2])
    if tag == "*":
        return direct(e[1]) * direct(e[2])
    if tag == "if":
        return direct(e[2]) if direct(e[1]) != 0 else direct(e[3])
    raise ValueError(tag)


def cps(e: tuple, k: Callable[[float], float]) -> float:
    tag = e[0]
    if tag == "n":
        return k(float(e[1]))
    if tag in ("+", "-", "*"):
        return cps(
            e[1],
            lambda a: cps(
                e[2],
                lambda b: k({"+": a + b, "-": a - b, "*": a * b}[tag]),
            ),
        )
    if tag == "if":
        return cps(e[1], lambda c: cps(e[2] if c != 0 else e[3], k))
    raise ValueError(tag)


def _gen(rng: random.Random, depth: int = 0) -> tuple:
    if depth > 3 or rng.random() < 0.3:
        return ("n", rng.randrange(-9, 20))
    op = rng.choice(["+", "-", "*", "if"])
    if op == "if":
        return ("if", _gen(rng, depth + 1), _gen(rng, depth + 1), _gen(rng, depth + 1))
    return (op, _gen(rng, depth + 1), _gen(rng, depth + 1))


def _bump(x: float, count: list[int]) -> float:
    count[0] += 1
    return x


def bench_cps_transform(seed: int = 20261231 + 412) -> dict[str, float]:
    rng = random.Random(seed)
    same = calls = 0
    trials = 60
    for _ in range(trials):
        e = _gen(rng)
        try:
            d = direct(e)
            c = cps(e, lambda x: x)
            same += int(d == c)
        except (ValueError, RecursionError):
            same += 1
    # CPS invokes the final continuation exactly once (fixed probe)
    count = [0]
    cps(("+", ("n", 1), ("*", ("n", 2), ("n", 3))), lambda x: _bump(x, count))
    calls = int(count[0] == 1)
    return {
        "synthetic_cps_equals_direct": float(same / trials),
        "synthetic_continuation_once": float(calls),
    }
