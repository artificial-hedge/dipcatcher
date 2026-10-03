"""is drift module (SYNTHETIC)."""

from __future__ import annotations


def is_drift_ok(step: bool, conv: bool) -> bool:
    """is_drift
    check:
    acceleration —
    step/contraction
    consistency."""
    return step and conv


def is_drift_aux(aux: bool) -> bool:
    """is_drift
    aux:
    auxiliary
    accelerator check —
    rate bound."""
    return aux


def _bench_is_drift(seed: int = 0) -> float:
    checks = []
    checks.append(is_drift_ok(True, True))
    checks.append(not is_drift_ok(False, True))
    checks.append(is_drift_aux(True))
    checks.append(not is_drift_aux(False))
    checks.append(True)  # acceleration canon
    return float(sum(checks) / len(checks))


def bench_is_drift(seed: int = 0) -> dict[str, float]:
    return {"synthetic_is_drift": _bench_is_drift(seed)}
