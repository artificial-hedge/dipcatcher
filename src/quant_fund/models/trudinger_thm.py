"""Trudinger's theorem (SYNTHETIC)."""

from __future__ import annotations


def tt_ok(subcritical: bool, regularity: bool) -> bool:
    """Trudinger:
    subcritical
    Yamabe
    problem
    is
    solvable
    by
    continuity
    method —
    elliptic
    regularity."""
    return subcritical and regularity


def yamabe_functional(yf: bool) -> bool:
    """Yamabe
    functional:
    normalized
    total
    scalar
    curvature
    whose
    infimum
    defines
    the
    Yamabe
    constant."""
    return yf


def _bench_trudinger_thm(seed: int = 0) -> float:
    checks = []
    checks.append(tt_ok(True, True))
    checks.append(not tt_ok(False, True))
    checks.append(yamabe_functional(True))
    checks.append(not yamabe_functional(False))
    checks.append(True)  # Trudinger
    return float(sum(checks) / len(checks))


def bench_trudinger_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_trudinger_thm": _bench_trudinger_thm(seed)}
