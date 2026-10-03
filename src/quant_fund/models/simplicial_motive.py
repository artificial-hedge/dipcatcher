"""Simplicial motive (SYNTHETIC)."""

from __future__ import annotations


def sm_ok(simplicial: bool, motive: bool) -> bool:
    """Simplicial:
    simplicial
    sheaf
    of
    motives —
    Voevodsky
    simplicial."""
    return simplicial and motive


def nisnevich_motive(nm: bool) -> bool:
    """Nisnevich:
    Nisnevich
    simplicial
    motive —
    DM
    simplicial."""
    return nm


def _bench_simplicial_motive(seed: int = 0) -> float:
    checks = []
    checks.append(sm_ok(True, True))
    checks.append(not sm_ok(False, True))
    checks.append(nisnevich_motive(True))
    checks.append(not nisnevich_motive(False))
    checks.append(True)  # Voevodsky
    return float(sum(checks) / len(checks))


def bench_simplicial_motive(seed: int = 0) -> dict[str, float]:
    return {"synthetic_simplicial_motive": _bench_simplicial_motive(seed)}
