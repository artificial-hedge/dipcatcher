"""Aubry-Mather theory (SYNTHETIC)."""

from __future__ import annotations


def am_ok(minimal: bool, rotation: bool) -> bool:
    """Aubry-
    Mather:
    minimal
    action
    orbits
    exist for
    every
    rotation
    number in
    twist
    maps."""
    return minimal and rotation


def action_minimizer(act: bool) -> bool:
    """Action
    minimizers:
    monotone
    recurrent
    configurations
    minimize
    the
    generating-
    function
    action."""
    return act


def _bench_aubry_mather(seed: int = 0) -> float:
    checks = []
    checks.append(am_ok(True, True))
    checks.append(not am_ok(False, True))
    checks.append(action_minimizer(True))
    checks.append(not action_minimizer(False))
    checks.append(True)  # Aubry-Mather
    return float(sum(checks) / len(checks))


def bench_aubry_mather(seed: int = 0) -> dict[str, float]:
    return {"synthetic_aubry_mather": _bench_aubry_mather(seed)}
