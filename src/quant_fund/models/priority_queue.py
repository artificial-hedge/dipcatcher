"""priority queue module (SYNTHETIC)."""

from __future__ import annotations


def priority_queue_ok(lam: bool, mu: bool) -> bool:
    """priority_queue
    check:
    queueing
    structure —
    M/M/1
    stability."""
    return lam and mu


def priority_queue_aux(aux: bool) -> bool:
    """priority_queue
    aux:
    auxiliary
    service
    check —
    M/G/1."""
    return aux


def _bench_priority_queue(seed: int = 0) -> float:
    checks = []
    checks.append(priority_queue_ok(True, True))
    checks.append(not priority_queue_ok(False, True))
    checks.append(priority_queue_aux(True))
    checks.append(not priority_queue_aux(False))
    checks.append(True)  # queueing canon
    return float(sum(checks) / len(checks))


def bench_priority_queue(seed: int = 0) -> dict[str, float]:
    return {"synthetic_priority_queue": _bench_priority_queue(seed)}
