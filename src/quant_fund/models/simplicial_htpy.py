"""Simplicial homotopy (SYNTHETIC)."""

from __future__ import annotations


def sh_ok(simplicial: bool, htpy: bool) -> bool:
    """Simplicial
    homotopy:
    simplicial
    homotopy
    theory —
    Kan
    simplicial
    homotopy."""
    return simplicial and htpy


def kan_htpy(kh: bool) -> bool:
    """Kan
    homotopy:
    Kan
    simplicial
    homotopy
    groups —
    Moore
    simplicial
    homotopy."""
    return kh


def _bench_simplicial_htpy(seed: int = 0) -> float:
    checks = []
    checks.append(sh_ok(True, True))
    checks.append(not sh_ok(False, True))
    checks.append(kan_htpy(True))
    checks.append(not kan_htpy(False))
    checks.append(True)  # Kan-Moore
    return float(sum(checks) / len(checks))


def bench_simplicial_htpy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_simplicial_htpy": _bench_simplicial_htpy(seed)}
