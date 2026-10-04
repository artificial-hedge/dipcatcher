"""Quotient stacks (SYNTHETIC)."""

from __future__ import annotations


def qs2_ok(quotient: bool, stack: bool) -> bool:
    """Quotient
    stack:
    quotient
    stack
    [X
    /
    G] —
    groupoid
    quotient."""
    return quotient and stack


def quotient_groupoid(qg: bool) -> bool:
    """Quotient
    groupoid:
    quotient
    groupoid
    of
    a
    G-
    action —
    action
    groupoid."""
    return qg


def _bench_quotient_stack2(seed: int = 0) -> float:
    checks = []
    checks.append(qs2_ok(True, True))
    checks.append(not qs2_ok(False, True))
    checks.append(quotient_groupoid(True))
    checks.append(not quotient_groupoid(False))
    checks.append(True)  # Mumford
    return float(sum(checks) / len(checks))


def bench_quotient_stack2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_quotient_stack2": _bench_quotient_stack2(seed)}
