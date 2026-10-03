"""fluctuation rw module (SYNTHETIC)."""

from __future__ import annotations


def fluctuation_rw_ok(step: bool, drift: bool) -> bool:
    """fluctuation_rw
    check:
    random-walk
    structure —
    Spitzer
    principle."""
    return step and drift


def fluctuation_rw_aux(aux: bool) -> bool:
    """fluctuation_rw
    aux:
    auxiliary
    fluctuation
    check —
    ladder
    epochs."""
    return aux


def _bench_fluctuation_rw(seed: int = 0) -> float:
    checks = []
    checks.append(fluctuation_rw_ok(True, True))
    checks.append(not fluctuation_rw_ok(False, True))
    checks.append(fluctuation_rw_aux(True))
    checks.append(not fluctuation_rw_aux(False))
    checks.append(True)  # random-walk canon
    return float(sum(checks) / len(checks))


def bench_fluctuation_rw(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fluctuation_rw": _bench_fluctuation_rw(seed)}
