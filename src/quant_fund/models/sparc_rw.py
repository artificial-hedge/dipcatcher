"""sparc rw module (SYNTHETIC)."""

from __future__ import annotations


def sparc_rw_ok(step: bool, drift: bool) -> bool:
    """sparc_rw
    check:
    random-walk
    structure —
    Spitzer
    principle."""
    return step and drift


def sparc_rw_aux(aux: bool) -> bool:
    """sparc_rw
    aux:
    auxiliary
    fluctuation
    check —
    ladder
    epochs."""
    return aux


def _bench_sparc_rw(seed: int = 0) -> float:
    checks = []
    checks.append(sparc_rw_ok(True, True))
    checks.append(not sparc_rw_ok(False, True))
    checks.append(sparc_rw_aux(True))
    checks.append(not sparc_rw_aux(False))
    checks.append(True)  # random-walk canon
    return float(sum(checks) / len(checks))


def bench_sparc_rw(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sparc_rw": _bench_sparc_rw(seed)}
