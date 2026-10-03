"""Little intervals/discs operad C_n(k) discretized (SYNTHETIC)."""

from __future__ import annotations

from itertools import combinations


def configs_1d(k: int, slots: int = 8) -> list[tuple[int, ...]]:
    """Little-1-cubes: k disjoint sub-intervals of [0, slots) as sorted
    left endpoints with unit length; ordering is forced (pi_0 = S_k order)."""
    return list(combinations(range(slots - k + 1), k))


def configs_2d(k: int, grid: int = 3) -> list[tuple[tuple[int, int], ...]]:
    """Little squares on a coarse grid: k unit cells in a grid x grid box.
    Unlike 1-D, any selection of k cells is one connected configuration
    (the pi_0 is trivial for unordered configs in dim >= 2)."""
    cells = [(i, j) for i in range(grid) for j in range(grid)]
    return list(combinations(cells, k))


def orbit_size_1d(k: int, slots: int = 8) -> int:
    """Number of connected components of ordered configs in dimension 1:
    the k! orderings are rigidly separated."""
    import math

    return math.factorial(k)


def _bench_little_discs(seed: int = 0) -> float:
    checks = []
    # little intervals of 2 in [0,8): C(7,2) positions
    checks.append(len(configs_1d(2, 8)) == 21)
    # ordered configs in dim 1 split into k! orderings (components)
    checks.append(orbit_size_1d(3) == 6)
    # grid configs C(9,2) for 2 cells on 3x3
    checks.append(len(configs_2d(2, 3)) == 36)
    # E2 commutativity up to homotopy: in dim >= 2 any two ordered
    # configurations are connected by a path swapping labels, so the
    # unordered config space is connected -> orbit count 1 vs 1d's 2
    checks.append(orbit_size_1d(2) == 2)
    # disjointness invariant: all generated configs have distinct cells
    checks.append(all(len(set(c)) == 2 for c in configs_2d(2, 3)))
    checks.append(all(len(set(c)) == 2 for c in configs_1d(2, 8)))
    return float(sum(checks) / len(checks))


def bench_little_discs(seed: int = 0) -> dict[str, float]:
    return {"synthetic_little_discs": _bench_little_discs(seed)}
