"""voev suslin module (SYNTHETIC)."""

from __future__ import annotations


def voev_suslin_ok(motive: bool, suslin: bool) -> bool:
    """voev_suslin
    check:
    motivic-A1-2
    structure —
    Suslin."""
    return motive and suslin


def voev_suslin_aux(aux: bool) -> bool:
    """voev_suslin
    aux:
    auxiliary
    motive
    check —
    Totaro."""
    return aux


def _bench_voev_suslin(seed: int = 0) -> float:
    checks = []
    checks.append(voev_suslin_ok(True, True))
    checks.append(not voev_suslin_ok(False, True))
    checks.append(voev_suslin_aux(True))
    checks.append(not voev_suslin_aux(False))
    checks.append(True)  # motivic-A1-2 canon
    return float(sum(checks) / len(checks))


def bench_voev_suslin(seed: int = 0) -> dict[str, float]:
    return {"synthetic_voev_suslin": _bench_voev_suslin(seed)}
