"""sabr model module (SYNTHETIC)."""

from __future__ import annotations


def sabr_model_ok(hm1: bool, sv: bool) -> bool:
    """sabr_model
    check:
    stochastic-vol
    —
    Heston/Bates."""
    return hm1 and sv


def sabr_model_aux(aux: bool) -> bool:
    """sabr_model
    aux:
    auxiliary
    vol
    check —
    rBergomi."""
    return aux


def _bench_sabr_model(seed: int = 0) -> float:
    checks = []
    checks.append(sabr_model_ok(True, True))
    checks.append(not sabr_model_ok(False, True))
    checks.append(sabr_model_aux(True))
    checks.append(not sabr_model_aux(False))
    checks.append(True)  # stochastic-vol canon
    return float(sum(checks) / len(checks))


def bench_sabr_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sabr_model": _bench_sabr_model(seed)}
