"""Exponential objects B^A in FinSet + curry/eval adjunction (SYNTHETIC)."""

from __future__ import annotations

import itertools


def all_maps(a: int, b: int) -> list[tuple[int, ...]]:
    """B^A = set of all functions A->B."""
    return list(itertools.product(range(b), repeat=a))


def ev(a: int, b: int) -> dict[tuple[int, int], int]:
    """eval: B^A × A -> B on encoded pair (func_index, x)."""
    maps = all_maps(a, b)
    return {(i, x): maps[i][x] for i in range(len(maps)) for x in range(a)}


def curry(f: dict[tuple[int, int], int], a: int, b: int, c: int) -> list[int]:
    """f: A×B -> C curried to A -> C^B (func indices into all_maps(b,c))."""
    maps = all_maps(b, c)
    idx = {m: i for i, m in enumerate(maps)}
    return [idx[tuple(f[(x, y)] for y in range(b))] for x in range(a)]


def uncurry(g: list[int], a: int, b: int, c: int) -> dict[tuple[int, int], int]:
    """A -> C^B uncurried to A×B -> C."""
    maps = all_maps(b, c)
    return {(x, y): maps[g[x]][y] for x in range(a) for y in range(b)}


def _bench_exponential_obj(seed: int = 0) -> float:
    checks = []
    checks.append(len(all_maps(3, 2)) == 8)
    e = ev(2, 3)
    checks.append(len(e) == 9 * 2)
    # curry/uncurry roundtrip on f: 2×2 -> 2
    f = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}
    g = curry(f, 2, 2, 2)
    checks.append(uncurry(g, 2, 2, 2) == f)
    # exponential universal: eval∘(curry f × id) = f
    maps = all_maps(2, 2)
    ok = all(maps[g[x]][y] == f[(x, y)] for x in range(2) for y in range(2))
    checks.append(ok)
    return sum(checks) / len(checks)


def bench_exponential_obj(seed: int = 0) -> dict[str, float]:
    return {"synthetic_exponential_obj": _bench_exponential_obj(seed)}
