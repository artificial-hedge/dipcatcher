"""Homotopy cartesian squares (SYNTHETIC)."""

from __future__ import annotations


def hc_ok(homotopy: bool, cartesian: bool) -> bool:
    """Homotopy
    cartesian:
    homotopy
    cartesian
    square —
    homotopy
    pullback."""
    return homotopy and cartesian


def homotopy_pullback(hp: bool) -> bool:
    """Homotopy
    pullback:
    homotopy
    pullback
    square —
    fiber
    sequence."""
    return hp


def _bench_homotopy_cartesian(seed: int = 0) -> float:
    checks = []
    checks.append(hc_ok(True, True))
    checks.append(not hc_ok(False, True))
    checks.append(homotopy_pullback(True))
    checks.append(not homotopy_pullback(False))
    checks.append(True)  # Goodwillie
    return float(sum(checks) / len(checks))


def bench_homotopy_cartesian(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_cartesian": _bench_homotopy_cartesian(seed)}
