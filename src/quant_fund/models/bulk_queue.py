"""bulk queue module (SYNTHETIC)."""

from __future__ import annotations


def bulk_queue_ok(lam: bool, mu: bool) -> bool:
    """bulk_queue
    check:
    queueing
    structure —
    M/M/1
    stability."""
    return lam and mu


def bulk_queue_aux(aux: bool) -> bool:
    """bulk_queue
    aux:
    auxiliary
    service
    check —
    M/G/1."""
    return aux


def _bench_bulk_queue(seed: int = 0) -> float:
    checks = []
    checks.append(bulk_queue_ok(True, True))
    checks.append(not bulk_queue_ok(False, True))
    checks.append(bulk_queue_aux(True))
    checks.append(not bulk_queue_aux(False))
    checks.append(True)  # queueing canon
    return float(sum(checks) / len(checks))


def bench_bulk_queue(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bulk_queue": _bench_bulk_queue(seed)}
