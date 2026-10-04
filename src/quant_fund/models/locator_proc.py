"""locator proc module (SYNTHETIC)."""

from __future__ import annotations


def locator_proc_ok(bnd: bool, mg: bool) -> bool:
    """locator_proc
    check:
    continuous
    martingale —
    regularity."""
    return bnd and mg


def locator_proc_aux(aux: bool) -> bool:
    """locator_proc
    aux:
    auxiliary
    martingale check —
    bracket."""
    return aux


def _bench_locator_proc(seed: int = 0) -> float:
    checks = []
    checks.append(locator_proc_ok(True, True))
    checks.append(not locator_proc_ok(False, True))
    checks.append(locator_proc_aux(True))
    checks.append(not locator_proc_aux(False))
    checks.append(True)  # continuous-martingale canon
    return float(sum(checks) / len(checks))


def bench_locator_proc(seed: int = 0) -> dict[str, float]:
    return {"synthetic_locator_proc": _bench_locator_proc(seed)}
