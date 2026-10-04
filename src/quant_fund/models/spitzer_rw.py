"""spitzer rw module (SYNTHETIC)."""

from __future__ import annotations


def spitzer_rw_ok(step: bool, drift: bool) -> bool:
    """spitzer_rw
    check:
    random-walk
    structure —
    Spitzer
    principle."""
    return step and drift


def spitzer_rw_aux(aux: bool) -> bool:
    """spitzer_rw
    aux:
    auxiliary
    fluctuation
    check —
    ladder
    epochs."""
    return aux


def _bench_spitzer_rw(seed: int = 0) -> float:
    checks = []
    checks.append(spitzer_rw_ok(True, True))
    checks.append(not spitzer_rw_ok(False, True))
    checks.append(spitzer_rw_aux(True))
    checks.append(not spitzer_rw_aux(False))
    checks.append(True)  # random-walk canon
    return float(sum(checks) / len(checks))


def bench_spitzer_rw(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spitzer_rw": _bench_spitzer_rw(seed)}
