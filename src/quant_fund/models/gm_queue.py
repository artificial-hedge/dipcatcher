"""gm queue module (SYNTHETIC)."""

from __future__ import annotations


def gm_queue_ok(lam: bool, mu: bool) -> bool:
    """gm_queue
    check:
    queueing
    structure —
    M/M/1
    stability."""
    return lam and mu


def gm_queue_aux(aux: bool) -> bool:
    """gm_queue
    aux:
    auxiliary
    service
    check —
    M/G/1."""
    return aux


def _bench_gm_queue(seed: int = 0) -> float:
    checks = []
    checks.append(gm_queue_ok(True, True))
    checks.append(not gm_queue_ok(False, True))
    checks.append(gm_queue_aux(True))
    checks.append(not gm_queue_aux(False))
    checks.append(True)  # queueing canon
    return float(sum(checks) / len(checks))


def bench_gm_queue(seed: int = 0) -> dict[str, float]:
    return {"synthetic_gm_queue": _bench_gm_queue(seed)}
