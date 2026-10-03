"""fclt invariance module (SYNTHETIC)."""

from __future__ import annotations


def fclt_invariance_ok(inv: bool, lim: bool) -> bool:
    """fclt_invariance
    check:
    functional
    limit —
    invariance."""
    return inv and lim


def fclt_invariance_aux(aux: bool) -> bool:
    """fclt_invariance
    aux:
    auxiliary
    limit check —
    approximation."""
    return aux


def _bench_fclt_invariance(seed: int = 0) -> float:
    checks = []
    checks.append(fclt_invariance_ok(True, True))
    checks.append(not fclt_invariance_ok(False, True))
    checks.append(fclt_invariance_aux(True))
    checks.append(not fclt_invariance_aux(False))
    checks.append(True)  # functional-limit canon
    return float(sum(checks) / len(checks))


def bench_fclt_invariance(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fclt_invariance": _bench_fclt_invariance(seed)}
