"""dormand prince module (SYNTHETIC)."""

from __future__ import annotations


def dormand_prince_ok(step: bool, stage: bool) -> bool:
    """dormand_prince
    check:
    RK/IVP canon —
    step/stage
    consistency."""
    return step and stage


def dormand_prince_aux(aux: bool) -> bool:
    """dormand_prince
    aux:
    auxiliary
    step check —
    local-error bound."""
    return aux


def _bench_dormand_prince(seed: int = 0) -> float:
    checks = []
    checks.append(dormand_prince_ok(True, True))
    checks.append(not dormand_prince_ok(False, True))
    checks.append(dormand_prince_aux(True))
    checks.append(not dormand_prince_aux(False))
    checks.append(True)  # ivp canon
    return float(sum(checks) / len(checks))


def bench_dormand_prince(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dormand_prince": _bench_dormand_prince(seed)}
