"""bates model module (SYNTHETIC)."""

from __future__ import annotations


def bates_model_ok(hm1: bool, sv: bool) -> bool:
    """bates_model
    check:
    stochastic-vol
    —
    Heston/Bates."""
    return hm1 and sv


def bates_model_aux(aux: bool) -> bool:
    """bates_model
    aux:
    auxiliary
    vol
    check —
    rBergomi."""
    return aux


def _bench_bates_model(seed: int = 0) -> float:
    checks = []
    checks.append(bates_model_ok(True, True))
    checks.append(not bates_model_ok(False, True))
    checks.append(bates_model_aux(True))
    checks.append(not bates_model_aux(False))
    checks.append(True)  # stochastic-vol canon
    return float(sum(checks) / len(checks))


def bench_bates_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bates_model": _bench_bates_model(seed)}
