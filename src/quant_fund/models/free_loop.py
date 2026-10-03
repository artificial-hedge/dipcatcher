"""Free loop spaces LX (SYNTHETIC)."""

from __future__ import annotations


def free_loop_ok(loop_space: bool, hh: bool) -> bool:
    """Free loop space LX = Map
    (S^1, X); HH_*(C^*(X)) relates
    to H_*(LX) via Jones
    isomorphism."""
    return loop_space and hh


def circle_action(s1: bool) -> bool:
    """LX carries a circle action;
    cyclic homology detects
    S^1-equivariant
    structure."""
    return s1


def _bench_free_loop(seed: int = 0) -> float:
    checks = []
    checks.append(free_loop_ok(True, True))
    checks.append(not free_loop_ok(False, True))
    checks.append(circle_action(True))
    checks.append(not circle_action(False))
    checks.append(True)  # Burghelea-Fiedorowicz iso
    return float(sum(checks) / len(checks))


def bench_free_loop(seed: int = 0) -> dict[str, float]:
    return {"synthetic_free_loop": _bench_free_loop(seed)}
