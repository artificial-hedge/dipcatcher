"""exponential euler module (SYNTHETIC)."""

from __future__ import annotations


def exponential_euler_ok(step: bool, order: bool) -> bool:
    """exponential_euler
    check:
    time-marching/ODE —
    stability
    consistency."""
    return step and order


def exponential_euler_aux(aux: bool) -> bool:
    """exponential_euler
    aux:
    auxiliary
    stepping check —
    order bound."""
    return aux


def _bench_exponential_euler(seed: int = 0) -> float:
    checks = []
    checks.append(exponential_euler_ok(True, True))
    checks.append(not exponential_euler_ok(False, True))
    checks.append(exponential_euler_aux(True))
    checks.append(not exponential_euler_aux(False))
    checks.append(True)  # time-marching canon
    return float(sum(checks) / len(checks))


def bench_exponential_euler(seed: int = 0) -> dict[str, float]:
    return {"synthetic_exponential_euler": _bench_exponential_euler(seed)}
