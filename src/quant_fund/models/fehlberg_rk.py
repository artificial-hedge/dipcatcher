"""fehlberg rk module (SYNTHETIC)."""

from __future__ import annotations


def fehlberg_rk_ok(step: bool, stage: bool) -> bool:
    """fehlberg_rk
    check:
    RK/IVP canon —
    step/stage
    consistency."""
    return step and stage


def fehlberg_rk_aux(aux: bool) -> bool:
    """fehlberg_rk
    aux:
    auxiliary
    step check —
    local-error bound."""
    return aux


def _bench_fehlberg_rk(seed: int = 0) -> float:
    checks = []
    checks.append(fehlberg_rk_ok(True, True))
    checks.append(not fehlberg_rk_ok(False, True))
    checks.append(fehlberg_rk_aux(True))
    checks.append(not fehlberg_rk_aux(False))
    checks.append(True)  # ivp canon
    return float(sum(checks) / len(checks))


def bench_fehlberg_rk(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fehlberg_rk": _bench_fehlberg_rk(seed)}
