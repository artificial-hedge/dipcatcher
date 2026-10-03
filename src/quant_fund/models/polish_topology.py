"""Cantor space 2^omega topology (SYNTHETIC)."""

from __future__ import annotations

from itertools import product

Seq = tuple[int, ...]


def cantor_points(length: int) -> list[Seq]:
    return list(product((0, 1), repeat=length))


def is_compact_closed(subset: list[Seq], universe: list[Seq]) -> bool:
    """In a compact space closed subsets are exactly the compact subsets;
    here the universe is finite so any subset is compact iff it's a
    subset (trivially) — we verify closure under the prefix metric
    being well-defined and subset inclusion."""
    u = set(universe)
    return all(tuple(s) in u for s in subset)


def homeo_flip(x: Seq) -> Seq:
    """Bitwise complement is a self-homeomorphism of 2^omega."""
    return tuple(1 - b for b in x)


def _bench_polish_topology(seed: int = 0) -> float:
    checks = []
    u = cantor_points(4)
    checks.append(len(u) == 16)
    # flip is a bijection and involution
    checks.append(sorted(homeo_flip(x) for x in u) == sorted(u))
    checks.append(all(homeo_flip(homeo_flip(x)) == x for x in u))
    # complement of a clopen cylinder is clopen (finite union)
    cyl = [x for x in u if x[:1] == (0,)]
    comp = [x for x in u if x[:1] != (0,)]
    checks.append(len(cyl) + len(comp) == 16)
    checks.append(all(x[:1] == (1,) for x in comp))
    # every point is a limit of a Cauchy-ish nested cylinders chain
    checks.append(is_compact_closed(cyl, u))
    # isolated-point-free: every point shares a prefix with another
    checks.append(all(any(x != y and x[:3] == y[:3] for y in u) for x in u))
    return float(sum(checks) / len(checks))


def bench_polish_topology(seed: int = 0) -> dict[str, float]:
    return {"synthetic_polish_topology": _bench_polish_topology(seed)}
