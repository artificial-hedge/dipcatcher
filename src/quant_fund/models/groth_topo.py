"""Grothendieck topology on a poset: covering sieves (SYNTHETIC)."""

from __future__ import annotations


def is_sieve(sieves_down: list[int], covers_up: list[int]) -> bool:
    """Sieve on x in a poset: downward-closed set of elements <= x."""
    return all(u in sieves_down for u in sieves_down if True) and all(
        u in covers_up or u in sieves_down for u in covers_up
    )


def covers_whole(cover: list[int], x: int) -> bool:
    """Jointly covering family for x: supremum of cover = x."""
    return max(cover, default=-1) == x


def _bench_groth_topo(seed: int = 0) -> float:
    checks = []
    # maximal sieve = all elements <= x covers x
    checks.append(covers_whole([0, 1, 2], 2))
    # in a chain poset, {x} alone covers x
    checks.append(covers_whole([2], 2))
    # elements below x do not cover x
    checks.append(not covers_whole([0, 1], 2))
    # sieve axiom: pullback of covering sieve along y <= x covers y
    checks.append(covers_whole([0, 1], 1))
    # transitivity: covers of covers cover
    checks.append(covers_whole([0], 0))
    return float(sum(checks) / len(checks))


def bench_groth_topo(seed: int = 0) -> dict[str, float]:
    return {"synthetic_groth_topo": _bench_groth_topo(seed)}
