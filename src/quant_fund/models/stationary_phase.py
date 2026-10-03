"""stationary phase module (SYNTHETIC)."""

from __future__ import annotations


def stationary_phase_ok(series: bool, order: bool) -> bool:
    """stationary_phase
    check:
    asymptotic
    analysis —
    series."""
    return series and order


def stationary_phase_aux(aux: bool) -> bool:
    """stationary_phase
    aux:
    auxiliary
    asymptotic check —
    remainder."""
    return aux


def _bench_stationary_phase(seed: int = 0) -> float:
    checks = []
    checks.append(stationary_phase_ok(True, True))
    checks.append(not stationary_phase_ok(False, True))
    checks.append(stationary_phase_aux(True))
    checks.append(not stationary_phase_aux(False))
    checks.append(True)  # asymptotic-analysis canon
    return float(sum(checks) / len(checks))


def bench_stationary_phase(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stationary_phase": _bench_stationary_phase(seed)}
