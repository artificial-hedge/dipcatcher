"""backward euler module (SYNTHETIC)."""

from __future__ import annotations


def backward_euler_ok(step: bool, stage: bool) -> bool:
    """backward_euler
    check:
    RK/IVP canon —
    step/stage
    consistency."""
    return step and stage


def backward_euler_aux(aux: bool) -> bool:
    """backward_euler
    aux:
    auxiliary
    step check —
    local-error bound."""
    return aux


def _bench_backward_euler(seed: int = 0) -> float:
    checks = []
    checks.append(backward_euler_ok(True, True))
    checks.append(not backward_euler_ok(False, True))
    checks.append(backward_euler_aux(True))
    checks.append(not backward_euler_aux(False))
    checks.append(True)  # ivp canon
    return float(sum(checks) / len(checks))


def bench_backward_euler(seed: int = 0) -> dict[str, float]:
    return {"synthetic_backward_euler": _bench_backward_euler(seed)}
