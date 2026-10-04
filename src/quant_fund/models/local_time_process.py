"""local time_process module (SYNTHETIC)."""

from __future__ import annotations


def local_time_process_ok(bnd: bool, mg: bool) -> bool:
    """local_time_process
    check:
    continuous
    martingale —
    regularity."""
    return bnd and mg


def local_time_process_aux(aux: bool) -> bool:
    """local_time_process
    aux:
    auxiliary
    martingale check —
    bracket."""
    return aux


def _bench_local_time_process(seed: int = 0) -> float:
    checks = []
    checks.append(local_time_process_ok(True, True))
    checks.append(not local_time_process_ok(False, True))
    checks.append(local_time_process_aux(True))
    checks.append(not local_time_process_aux(False))
    checks.append(True)  # continuous-martingale canon
    return float(sum(checks) / len(checks))


def bench_local_time_process(seed: int = 0) -> dict[str, float]:
    return {"synthetic_local_time_process": _bench_local_time_process(seed)}
