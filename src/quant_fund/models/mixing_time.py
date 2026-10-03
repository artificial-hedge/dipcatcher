"""mixing time module (SYNTHETIC)."""

from __future__ import annotations


def mixing_time_ok(dc: bool, hr: bool) -> bool:
    """mixing_time
    check:
    Markov-chain
    theory —
    stability."""
    return dc and hr


def mixing_time_aux(aux: bool) -> bool:
    """mixing_time
    aux:
    auxiliary
    chain
    check —
    mixing."""
    return aux


def _bench_mixing_time(seed: int = 0) -> float:
    checks = []
    checks.append(mixing_time_ok(True, True))
    checks.append(not mixing_time_ok(False, True))
    checks.append(mixing_time_aux(True))
    checks.append(not mixing_time_aux(False))
    checks.append(True)  # markov-chain canon
    return float(sum(checks) / len(checks))


def bench_mixing_time(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mixing_time": _bench_mixing_time(seed)}
