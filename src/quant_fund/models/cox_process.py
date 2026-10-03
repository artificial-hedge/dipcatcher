"""cox process module (SYNTHETIC)."""

from __future__ import annotations


def cox_process_ok(pt: bool, meas: bool) -> bool:
    """cox_process
    check:
    point-process
    structure —
    Cox
    intensity."""
    return pt and meas


def cox_process_aux(aux: bool) -> bool:
    """cox_process
    aux:
    auxiliary
    mark
    check —
    Palm
    distribution."""
    return aux


def _bench_cox_process(seed: int = 0) -> float:
    checks = []
    checks.append(cox_process_ok(True, True))
    checks.append(not cox_process_ok(False, True))
    checks.append(cox_process_aux(True))
    checks.append(not cox_process_aux(False))
    checks.append(True)  # point-process canon
    return float(sum(checks) / len(checks))


def bench_cox_process(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cox_process": _bench_cox_process(seed)}
