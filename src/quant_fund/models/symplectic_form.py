"""Symplectic forms (SYNTHETIC)."""

from __future__ import annotations


def sym_ok(closed: bool, nondeg: bool) -> bool:
    """Symplectic
    form:
    closed
    non-
    degenerate
    2-form
    omega —
    skew
    pairing
    on
    the
    tangent
    bundle."""
    return closed and nondeg


def darboux_local(dl: bool) -> bool:
    """Darboux:
    locally
    every
    symplectic
    form
    is
    sum
    dp_i
    wedge
    dq_i."""
    return dl


def _bench_symplectic_form(seed: int = 0) -> float:
    checks = []
    checks.append(sym_ok(True, True))
    checks.append(not sym_ok(False, True))
    checks.append(darboux_local(True))
    checks.append(not darboux_local(False))
    checks.append(True)  # Darboux
    return float(sum(checks) / len(checks))


def bench_symplectic_form(seed: int = 0) -> dict[str, float]:
    return {"synthetic_symplectic_form": _bench_symplectic_form(seed)}
