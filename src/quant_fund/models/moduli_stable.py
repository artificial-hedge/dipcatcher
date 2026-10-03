"""Moduli of stable pointed curves (SYNTHETIC)."""

from __future__ import annotations

from itertools import combinations


def mgn_dim(g: int, n: int) -> int:
    """dim M_{g,n} = dim of the moduli space = 3g - 3 + n."""
    return 3 * g - 3 + n


def is_stable(
    markings_per_comp: list[int], nodes_per_comp: list[int], genus_per_comp: list[int]
) -> bool:
    """Stability: every genus-0 component has >= 3 special points
    (marks + nodes); every genus-1 component has >= 1."""
    return all(
        (g == 0 and m + nd >= 3) or (g == 1 and m + nd >= 1) or g >= 2
        for g, m, nd in zip(genus_per_comp, markings_per_comp, nodes_per_comp, strict=True)
    )


def boundary_strata_n4() -> int:
    """Nodal stable curves in M-bar_{0,4}: 2 components meeting in 1 node,
    markings split 2+2 between them. Unordered partitions of {1,2,3,4}
    into two pairs = 3."""
    marks = {0, 1, 2, 3}
    count = 0
    seen = set()
    for first in combinations(marks, 2):
        second = tuple(sorted(marks - set(first)))
        key = frozenset({frozenset(first), frozenset(second)})
        if key not in seen:
            seen.add(key)
            count += 1
    return count


def _bench_moduli_stable(seed: int = 0) -> float:
    checks = []
    # dim M_{0,4} = 1, M_{1,1} = 1, M_{2,0} = 3, M_{0,5} = 2, M_{3,0} = 6
    checks.append(mgn_dim(0, 4) == 1)
    checks.append(mgn_dim(1, 1) == 1)
    checks.append(mgn_dim(2, 0) == 3)
    checks.append(mgn_dim(0, 5) == 2)
    checks.append(mgn_dim(3, 0) == 6)
    # stability checks
    checks.append(is_stable([4], [0], [0]))  # smooth P1 with 4 marks
    checks.append(is_stable([2, 2], [1, 1], [0, 0]))  # 2+2 boundary
    checks.append(not is_stable([1, 3], [1, 1], [0, 0]))  # 1+3 split unstable
    checks.append(not is_stable([1], [0], [0]))  # smooth P1, 1 mark unstable
    checks.append(is_stable([0], [0], [2]))  # genus 2 needs nothing
    # 3 boundary strata of M_{0,4}
    checks.append(boundary_strata_n4() == 3)
    return float(sum(checks) / len(checks))


def bench_moduli_stable(seed: int = 0) -> dict[str, float]:
    return {"synthetic_moduli_stable": _bench_moduli_stable(seed)}
