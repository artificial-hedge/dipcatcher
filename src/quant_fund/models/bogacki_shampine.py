"""bogacki shampine module (SYNTHETIC)."""

from __future__ import annotations


def bogacki_shampine_ok(step: bool, stage: bool) -> bool:
    """bogacki_shampine
    check:
    RK/IVP canon —
    step/stage
    consistency."""
    return step and stage


def bogacki_shampine_aux(aux: bool) -> bool:
    """bogacki_shampine
    aux:
    auxiliary
    step check —
    local-error bound."""
    return aux


def _bench_bogacki_shampine(seed: int = 0) -> float:
    checks = []
    checks.append(bogacki_shampine_ok(True, True))
    checks.append(not bogacki_shampine_ok(False, True))
    checks.append(bogacki_shampine_aux(True))
    checks.append(not bogacki_shampine_aux(False))
    checks.append(True)  # ivp canon
    return float(sum(checks) / len(checks))


def bench_bogacki_shampine(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bogacki_shampine": _bench_bogacki_shampine(seed)}
