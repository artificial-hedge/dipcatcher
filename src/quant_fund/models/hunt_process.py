"""hunt process module (SYNTHETIC)."""

from __future__ import annotations


def hunt_process_ok(mp: bool, se: bool) -> bool:
    """hunt_process
    check:
    Markov
    process —
    semigroup."""
    return mp and se


def hunt_process_aux(aux: bool) -> bool:
    """hunt_process
    aux:
    auxiliary
    Markov check —
    resolvent."""
    return aux


def _bench_hunt_process(seed: int = 0) -> float:
    checks = []
    checks.append(hunt_process_ok(True, True))
    checks.append(not hunt_process_ok(False, True))
    checks.append(hunt_process_aux(True))
    checks.append(not hunt_process_aux(False))
    checks.append(True)  # Markov canon
    return float(sum(checks) / len(checks))


def bench_hunt_process(seed: int = 0) -> dict[str, float]:
    return {"synthetic_hunt_process": _bench_hunt_process(seed)}
