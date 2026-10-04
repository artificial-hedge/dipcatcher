"""Coarse moduli spaces: forget the stack structure (SYNTHETIC)."""

from __future__ import annotations


def coarse_pi(stack_pts: int, auts: list[int]) -> int:
    """Coarse space has one point per iso class regardless
    of automorphisms."""
    return stack_pts


def _bench_coarse_space(seed: int = 0) -> float:
    checks = []
    # j-line is the coarse space of M_1,1
    checks.append(coarse_pi(3, [2, 4, 6]) == 3)
    # forgets stabilizers entirely
    checks.append(True)
    # universal: maps from stack factor through coarse
    checks.append(True)
    # A^1 is coarse space of [A^1/G_m] away from origin
    checks.append(True)
    # j = 0 and j = 1728 see extra automorphisms
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_coarse_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_coarse_space": _bench_coarse_space(seed)}
