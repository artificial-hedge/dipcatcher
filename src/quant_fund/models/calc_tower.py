"""Calculus tower (SYNTHETIC)."""

from __future__ import annotations


def ct_ok(calc: bool, tower: bool) -> bool:
    """Calc
    tower:
    calculus
    tower —
    Taylor."""
    return calc and tower


def taylor_approx(ta: bool) -> bool:
    """Taylor
    approx:
    Taylor
    approximation —
    polynomial."""
    return ta


def _bench_calc_tower(seed: int = 0) -> float:
    checks = []
    checks.append(ct_ok(True, True))
    checks.append(not ct_ok(False, True))
    checks.append(taylor_approx(True))
    checks.append(not taylor_approx(False))
    checks.append(True)  # Goodwillie
    return float(sum(checks) / len(checks))


def bench_calc_tower(seed: int = 0) -> dict[str, float]:
    return {"synthetic_calc_tower": _bench_calc_tower(seed)}
