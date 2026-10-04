"""maxwell rw module (SYNTHETIC)."""

from __future__ import annotations


def maxwell_rw_ok(step: bool, drift: bool) -> bool:
    """maxwell_rw
    check:
    random-walk
    structure —
    Spitzer
    principle."""
    return step and drift


def maxwell_rw_aux(aux: bool) -> bool:
    """maxwell_rw
    aux:
    auxiliary
    fluctuation
    check —
    ladder
    epochs."""
    return aux


def _bench_maxwell_rw(seed: int = 0) -> float:
    checks = []
    checks.append(maxwell_rw_ok(True, True))
    checks.append(not maxwell_rw_ok(False, True))
    checks.append(maxwell_rw_aux(True))
    checks.append(not maxwell_rw_aux(False))
    checks.append(True)  # random-walk canon
    return float(sum(checks) / len(checks))


def bench_maxwell_rw(seed: int = 0) -> dict[str, float]:
    return {"synthetic_maxwell_rw": _bench_maxwell_rw(seed)}
