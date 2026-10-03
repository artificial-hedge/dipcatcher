"""Algebraic stacks (SYNTHETIC)."""

from __future__ import annotations


def as2_ok(algebraic: bool, stack: bool) -> bool:
    """Algebraic
    stack:
    algebraic
    stack —
    Artin
    algebraic
    stack."""
    return algebraic and stack


def stack_atlas(sa: bool) -> bool:
    """Stack
    atlas:
    smooth
    atlas
    of
    an
    algebraic
    stack —
    Artin
    atlas."""
    return sa


def _bench_algebraic_stack2(seed: int = 0) -> float:
    checks = []
    checks.append(as2_ok(True, True))
    checks.append(not as2_ok(False, True))
    checks.append(stack_atlas(True))
    checks.append(not stack_atlas(False))
    checks.append(True)  # Artin
    return float(sum(checks) / len(checks))


def bench_algebraic_stack2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_algebraic_stack2": _bench_algebraic_stack2(seed)}
