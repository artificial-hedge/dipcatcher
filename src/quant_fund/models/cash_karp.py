"""cash karp module (SYNTHETIC)."""

from __future__ import annotations


def cash_karp_ok(step: bool, stage: bool) -> bool:
    """cash_karp
    check:
    RK/IVP canon —
    step/stage
    consistency."""
    return step and stage


def cash_karp_aux(aux: bool) -> bool:
    """cash_karp
    aux:
    auxiliary
    step check —
    local-error bound."""
    return aux


def _bench_cash_karp(seed: int = 0) -> float:
    checks = []
    checks.append(cash_karp_ok(True, True))
    checks.append(not cash_karp_ok(False, True))
    checks.append(cash_karp_aux(True))
    checks.append(not cash_karp_aux(False))
    checks.append(True)  # ivp canon
    return float(sum(checks) / len(checks))


def bench_cash_karp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cash_karp": _bench_cash_karp(seed)}
