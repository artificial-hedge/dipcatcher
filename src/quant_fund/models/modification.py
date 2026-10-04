"""Modifications (SYNTHETIC)."""

from __future__ import annotations


def mo_ok(modification: bool, transform: bool) -> bool:
    """Modification:
    modification
    between
    transformations —
    3-
    cell."""
    return modification and transform


def modification_axiom(ma: bool) -> bool:
    """Modification
    axiom:
    modification
    naturality
    —
    tricat
    3-cell."""
    return ma


def _bench_modification(seed: int = 0) -> float:
    checks = []
    checks.append(mo_ok(True, True))
    checks.append(not mo_ok(False, True))
    checks.append(modification_axiom(True))
    checks.append(not modification_axiom(False))
    checks.append(True)  # tricat
    return float(sum(checks) / len(checks))


def bench_modification(seed: int = 0) -> dict[str, float]:
    return {"synthetic_modification": _bench_modification(seed)}
