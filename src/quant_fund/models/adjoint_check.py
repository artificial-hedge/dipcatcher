"""Adjunction verification: product ⊣ exponential in FinSet (SYNTHETIC)."""

from __future__ import annotations

import itertools


def hom_ab(a: int, b: int) -> list[tuple[int, ...]]:
    return list(itertools.product(range(b), repeat=a))


def hom_prod_to_c(a: int, b: int, c: int) -> list[tuple[int, ...]]:
    """Hom(A×B, C)."""
    return list(itertools.product(range(c), repeat=a * b))


def curry_hom(f: tuple[int, ...], a: int, b: int, c: int) -> tuple[int, ...]:
    """Hom(A×B,C) -> Hom(A, C^B): g(x) = index of y->f(x*b+y) among C^B."""
    maps = hom_ab(b, c)
    idx = {m: i for i, m in enumerate(maps)}
    return tuple(idx[tuple(f[x * b + y] for y in range(b))] for x in range(a))


def uncurry_hom(g: tuple[int, ...], a: int, b: int, c: int) -> tuple[int, ...]:
    """Hom(A, C^B) -> Hom(A×B, C)."""
    maps = hom_ab(b, c)
    return tuple(maps[g[x]][y] for x in range(a) for y in range(b))


def _bench_adjoint_check(seed: int = 0) -> float:
    checks = []
    a, b, c = 2, 3, 2
    lhs = hom_prod_to_c(a, b, c)
    rhs = hom_ab(a, c**b)
    checks.append(len(lhs) == len(rhs))  # adjunction bijection cardinalities
    f = lhs[5]
    checks.append(uncurry_hom(curry_hom(f, a, b, c), a, b, c) == f)
    g = rhs[7]
    checks.append(curry_hom(uncurry_hom(g, a, b, c), a, b, c) == g)
    return sum(checks) / len(checks)


def bench_adjoint_check(seed: int = 0) -> dict[str, float]:
    return {"synthetic_adjoint_check": _bench_adjoint_check(seed)}
