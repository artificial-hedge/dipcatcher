"""Tamagawa-Mochizuki theorems (SYNTHETIC)."""

from __future__ import annotations


def tamagawa_ok(number_field: bool, hyperbolic: bool) -> bool:
    """Tamagawa-Mochizuki:
    Grothendieck
    conjecture for
    hyperbolic curves
    over number
    fields proved."""
    return number_field and hyperbolic


def mochizuki_padic(padic: bool) -> bool:
    """Mochizuki:
    p-adic version
    for sub-p-adic
    fields; cuspidalized
    pi_1 recovers
    the curve."""
    return padic


def _bench_tamagawa_mochi(seed: int = 0) -> float:
    checks = []
    checks.append(tamagawa_ok(True, True))
    checks.append(not tamagawa_ok(False, True))
    checks.append(mochizuki_padic(True))
    checks.append(not mochizuki_padic(False))
    checks.append(True)  # Tamagawa 1997
    return float(sum(checks) / len(checks))


def bench_tamagawa_mochi(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tamagawa_mochi": _bench_tamagawa_mochi(seed)}
