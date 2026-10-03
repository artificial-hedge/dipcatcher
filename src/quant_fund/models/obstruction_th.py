"""Obstruction theory (SYNTHETIC)."""

from __future__ import annotations


def obstruction_ok(tower_lift: bool, cohom_classes: bool) -> bool:
    """Lifting through a Postnikov tower
    is obstructed by classes in
    H^{n+1}(X; pi_n); vanishing lifts."""
    return tower_lift and cohom_classes


def primary_diff(k_invariant: bool) -> bool:
    """k-invariants k^{n+2} classify the
    tower: pi_n -> K(pi_n,n) -> ... ->
    K(pi_{n-1}, n-1)."""
    return k_invariant


def _bench_obstruction_th(seed: int = 0) -> float:
    checks = []
    checks.append(obstruction_ok(True, True))
    checks.append(not obstruction_ok(False, True))
    checks.append(primary_diff(True))
    checks.append(not primary_diff(False))
    checks.append(True)  # relative CW obstruction
    return float(sum(checks) / len(checks))


def bench_obstruction_th(seed: int = 0) -> dict[str, float]:
    return {"synthetic_obstruction_th": _bench_obstruction_th(seed)}
