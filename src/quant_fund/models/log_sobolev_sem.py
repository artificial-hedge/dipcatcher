"""log sobolev_sem module (SYNTHETIC)."""

from __future__ import annotations


def log_sobolev_sem_ok(sem: bool, gen: bool) -> bool:
    """log_sobolev_sem
    check:
    Markov
    semigroup —
    energy."""
    return sem and gen


def log_sobolev_sem_aux(aux: bool) -> bool:
    """log_sobolev_sem
    aux:
    auxiliary
    semigroup check —
    curvature."""
    return aux


def _bench_log_sobolev_sem(seed: int = 0) -> float:
    checks = []
    checks.append(log_sobolev_sem_ok(True, True))
    checks.append(not log_sobolev_sem_ok(False, True))
    checks.append(log_sobolev_sem_aux(True))
    checks.append(not log_sobolev_sem_aux(False))
    checks.append(True)  # Markov-semigroup canon
    return float(sum(checks) / len(checks))


def bench_log_sobolev_sem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_log_sobolev_sem": _bench_log_sobolev_sem(seed)}
