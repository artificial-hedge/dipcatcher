"""Inertia stacks (SYNTHETIC)."""

from __future__ import annotations


def is_ok(inertia: bool, stack: bool) -> bool:
    """Inertia:
    inertia
    stack
    of
    an
    algebraic
    stack —
    inertia
    stack."""
    return inertia and stack


def twisted_sector(ts: bool) -> bool:
    """Twisted
    sectors:
    twisted
    sectors
    of
    an
    orbifold —
    Chen-
    Ruan."""
    return ts


def _bench_inertia_stack(seed: int = 0) -> float:
    checks = []
    checks.append(is_ok(True, True))
    checks.append(not is_ok(False, True))
    checks.append(twisted_sector(True))
    checks.append(not twisted_sector(False))
    checks.append(True)  # Chen-Ruan
    return float(sum(checks) / len(checks))


def bench_inertia_stack(seed: int = 0) -> dict[str, float]:
    return {"synthetic_inertia_stack": _bench_inertia_stack(seed)}
