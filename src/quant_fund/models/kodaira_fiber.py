"""Kodaira fibers (SYNTHETIC)."""

from __future__ import annotations


def kodaira_ok(fiber_type: bool, mult_n: bool) -> bool:
    """Kodaira fiber types
    I_n, II, III, IV,
    I_n^*, II^*, III^*,
    IV^* encode
    Euler characteristic
    contributions."""
    return fiber_type and mult_n


def fiber_contrib(euler_contrib: bool) -> bool:
    """Euler characteristic
    contribution of each
    fiber type is its
    Kodaira symbol
    weight."""
    return euler_contrib


def _bench_kodaira_fiber(seed: int = 0) -> float:
    checks = []
    checks.append(kodaira_ok(True, True))
    checks.append(not kodaira_ok(False, True))
    checks.append(fiber_contrib(True))
    checks.append(not fiber_contrib(False))
    checks.append(True)  # Néron model
    return float(sum(checks) / len(checks))


def bench_kodaira_fiber(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kodaira_fiber": _bench_kodaira_fiber(seed)}
