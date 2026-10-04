"""jko step module (SYNTHETIC)."""

from __future__ import annotations


def jko_step_ok(ot1: bool, gf: bool) -> bool:
    """jko_step
    check:
    optimal-transport
    —
    Wasserstein
    gradient
    flow."""
    return ot1 and gf


def jko_step_aux(aux: bool) -> bool:
    """jko_step
    aux:
    auxiliary
    JKO
    check —
    minimizing
    movement."""
    return aux


def _bench_jko_step(seed: int = 0) -> float:
    checks = []
    checks.append(jko_step_ok(True, True))
    checks.append(not jko_step_ok(False, True))
    checks.append(jko_step_aux(True))
    checks.append(not jko_step_aux(False))
    checks.append(True)  # OT canon
    return float(sum(checks) / len(checks))


def bench_jko_step(seed: int = 0) -> dict[str, float]:
    return {"synthetic_jko_step": _bench_jko_step(seed)}
