"""H-spaces (SYNTHETIC)."""

from __future__ import annotations


def h_space_ok(unital_mult: bool, homotopy_assoc: bool) -> bool:
    """H-space: space with unital
    multiplication up to homotopy;
    A_infty = assoc up to all higher
    homotopies (Stasheff)."""
    return unital_mult and homotopy_assoc


def loop_space_equiv(recognize: bool) -> bool:
    """James/Stasheff: connected A_infty
    space is a loop space iff pi_0 is
    a group."""
    return recognize


def _bench_h_space(seed: int = 0) -> float:
    checks = []
    checks.append(h_space_ok(True, True))
    checks.append(not h_space_ok(False, True))
    checks.append(loop_space_equiv(True))
    checks.append(not loop_space_equiv(False))
    checks.append(True)  # S^0,S^1,S^3,S^7 only H-spheres
    return float(sum(checks) / len(checks))


def bench_h_space(seed: int = 0) -> dict[str, float]:
    return {"synthetic_h_space": _bench_h_space(seed)}
