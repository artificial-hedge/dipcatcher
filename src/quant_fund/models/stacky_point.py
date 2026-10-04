"""Stacky points (SYNTHETIC)."""

from __future__ import annotations


def sp_ok(stacky: bool, point: bool) -> bool:
    """Stacky
    point:
    stacky
    point
    with
    stabilizer —
    BG
    point."""
    return stacky and point


def classifying_stack(cs: bool) -> bool:
    """Classifying
    stack:
    classifying
    stack
    BG —
    classifying
    stacky
    point."""
    return cs


def _bench_stacky_point(seed: int = 0) -> float:
    checks = []
    checks.append(sp_ok(True, True))
    checks.append(not sp_ok(False, True))
    checks.append(classifying_stack(True))
    checks.append(not classifying_stack(False))
    checks.append(True)  # BG
    return float(sum(checks) / len(checks))


def bench_stacky_point(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stacky_point": _bench_stacky_point(seed)}
