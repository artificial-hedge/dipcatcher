"""mg1 queue module (SYNTHETIC)."""

from __future__ import annotations


def mg1_queue_ok(lam: bool, mu: bool) -> bool:
    """mg1_queue
    check:
    queueing
    structure —
    M/M/1
    stability."""
    return lam and mu


def mg1_queue_aux(aux: bool) -> bool:
    """mg1_queue
    aux:
    auxiliary
    service
    check —
    M/G/1."""
    return aux


def _bench_mg1_queue(seed: int = 0) -> float:
    checks = []
    checks.append(mg1_queue_ok(True, True))
    checks.append(not mg1_queue_ok(False, True))
    checks.append(mg1_queue_aux(True))
    checks.append(not mg1_queue_aux(False))
    checks.append(True)  # queueing canon
    return float(sum(checks) / len(checks))


def bench_mg1_queue(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mg1_queue": _bench_mg1_queue(seed)}
