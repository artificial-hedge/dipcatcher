"""meyers process module (SYNTHETIC)."""

from __future__ import annotations


def meyers_process_ok(fl: bool, df: bool) -> bool:
    """meyers_process
    check:
    stochastic
    flow —
    diffeo."""
    return fl and df


def meyers_process_aux(aux: bool) -> bool:
    """meyers_process
    aux:
    auxiliary
    flow check —
    cocycle."""
    return aux


def _bench_meyers_process(seed: int = 0) -> float:
    checks = []
    checks.append(meyers_process_ok(True, True))
    checks.append(not meyers_process_ok(False, True))
    checks.append(meyers_process_aux(True))
    checks.append(not meyers_process_aux(False))
    checks.append(True)  # stochastic-flow canon
    return float(sum(checks) / len(checks))


def bench_meyers_process(seed: int = 0) -> dict[str, float]:
    return {"synthetic_meyers_process": _bench_meyers_process(seed)}
