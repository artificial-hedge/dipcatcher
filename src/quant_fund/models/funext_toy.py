"""Function extensionality on finite domains.

funext: (∀x. f(x) = g(x)) -> f = g. On finite domains the hypothesis is
checked pointwise; when it holds we emit the canonical path between f
and g (extensional equality => definitional in the toy model).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_SEED = 20261231 + 1057


def funext(f: Callable[[Any], Any], g: Callable[[Any], Any], domain: list) -> tuple | None:
    """Returns a path tag ("funext", f=g) when pointwise equal, else None."""
    if all(f(x) == g(x) for x in domain):
        return ("funext", id(f) if f is g else "ext")
    return None


def pointwise_eq(f: Callable, g: Callable, domain: list) -> bool:
    return all(f(x) == g(x) for x in domain)


def apply_path(f: Callable, g: Callable, p: tuple | None, x):
    """If p witnesses f = g, applying g equals applying f."""
    if p is None:
        return g(x)
    return f(x)


def bench_funext_toy(seed: int = _SEED) -> dict[str, float]:
    del seed
    checks: list[bool] = []

    def f(x: int) -> int:
        return x + 1

    def g(x: int) -> int:
        return x + 1

    def h(x: int) -> int:
        return (x + 2) - 1

    def other(x: int) -> int:
        return x + 2

    def f2(x: int) -> int:
        return x if x < 5 else 999

    def g2(x: int) -> int:
        return x if x < 5 else -1

    dom = list(range(5))
    checks.append(funext(f, g, dom) is not None)
    # different implementations, same values
    checks.append(pointwise_eq(f, h, dom))
    # genuinely different function rejected
    checks.append(funext(f, other, dom) is None)
    # apply through the path
    p = funext(f, g, dom)
    checks.append(apply_path(f, g, p, 4) == 5)
    # restricted domain: agree on domain, differ outside -> still funext
    checks.append(funext(f2, g2, dom) is not None)
    return {"synthetic_funext_toy": float(sum(checks)) / len(checks)}
