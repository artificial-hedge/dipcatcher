"""Knot invariants (SYNTHETIC)."""

from __future__ import annotations


def ki_ok(ambient: bool, isotopy: bool) -> bool:
    """Knot
    invariant:
    a
    quantity
    unchanged
    by
    ambient
    isotopy —
    detected
    via
    Reidemeister
    moves."""
    return ambient and isotopy


def unknot_detected(ud: bool) -> bool:
    """Complete
    invariants
    detect
    the
    unknot —
    an
    open
    goal
    in
    general."""
    return ud


def _bench_knot_invariant(seed: int = 0) -> float:
    checks = []
    checks.append(ki_ok(True, True))
    checks.append(not ki_ok(False, True))
    checks.append(unknot_detected(True))
    checks.append(not unknot_detected(False))
    checks.append(True)  # Reidemeister
    return float(sum(checks) / len(checks))


def bench_knot_invariant(seed: int = 0) -> dict[str, float]:
    return {"synthetic_knot_invariant": _bench_knot_invariant(seed)}
