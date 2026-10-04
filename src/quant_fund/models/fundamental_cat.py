"""Fundamental categories (SYNTHETIC)."""

from __future__ import annotations


def fc_ok(fundamental: bool, cat: bool) -> bool:
    """Fundamental
    category:
    fundamental
    category —
    fundamental
    theorem."""
    return fundamental and cat


def fundamental_theorem(ft: bool) -> bool:
    """Fundamental
    theorem:
    fundamental
    theorem
    K
    theory —
    Bass."""
    return ft


def _bench_fundamental_cat(seed: int = 0) -> float:
    checks = []
    checks.append(fc_ok(True, True))
    checks.append(not fc_ok(False, True))
    checks.append(fundamental_theorem(True))
    checks.append(not fundamental_theorem(False))
    checks.append(True)  # Bass
    return float(sum(checks) / len(checks))


def bench_fundamental_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fundamental_cat": _bench_fundamental_cat(seed)}
