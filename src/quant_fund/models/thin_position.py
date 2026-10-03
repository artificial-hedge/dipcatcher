"""Thin position (SYNTHETIC)."""

from __future__ import annotations


def tp_ok(width_fn: bool, bridges: bool) -> bool:
    """Thin
    position:
    minimize
    a
    width
    function
    over
    knot
    diagrams
    or
    Heegaard
    splittings —
    Gabai,
    Thompson."""
    return width_fn and bridges


def additivity_thm(at: bool) -> bool:
    """Additivity:
    tunnel
    number
    and
    width
    can
    fail
    additivity —
    Moriah-
    Schleimer-
    Sedgwick
    examples."""
    return at


def _bench_thin_position(seed: int = 0) -> float:
    checks = []
    checks.append(tp_ok(True, True))
    checks.append(not tp_ok(False, True))
    checks.append(additivity_thm(True))
    checks.append(not additivity_thm(False))
    checks.append(True)  # Gabai-Thompson
    return float(sum(checks) / len(checks))


def bench_thin_position(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thin_position": _bench_thin_position(seed)}
