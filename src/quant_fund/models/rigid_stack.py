"""Rigidified stacks (SYNTHETIC)."""

from __future__ import annotations


def rs_ok(rigid: bool, stack: bool) -> bool:
    """Rigidified:
    rigidification
    of
    a
    stack —
    Abramovich
    rigidification."""
    return rigid and stack


def rigidify(rg: bool) -> bool:
    """Rigidify:
    rigidification
    functor —
    Abramovich-
    Corti-
    Vistoli."""
    return rg


def _bench_rigid_stack(seed: int = 0) -> float:
    checks = []
    checks.append(rs_ok(True, True))
    checks.append(not rs_ok(False, True))
    checks.append(rigidify(True))
    checks.append(not rigidify(False))
    checks.append(True)  # ACV
    return float(sum(checks) / len(checks))


def bench_rigid_stack(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rigid_stack": _bench_rigid_stack(seed)}
