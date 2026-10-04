"""heston model module (SYNTHETIC)."""

from __future__ import annotations


def heston_model_ok(hm1: bool, sv: bool) -> bool:
    """heston_model
    check:
    stochastic-vol
    —
    Heston/Bates."""
    return hm1 and sv


def heston_model_aux(aux: bool) -> bool:
    """heston_model
    aux:
    auxiliary
    vol
    check —
    rBergomi."""
    return aux


def _bench_heston_model(seed: int = 0) -> float:
    checks = []
    checks.append(heston_model_ok(True, True))
    checks.append(not heston_model_ok(False, True))
    checks.append(heston_model_aux(True))
    checks.append(not heston_model_aux(False))
    checks.append(True)  # stochastic-vol canon
    return float(sum(checks) / len(checks))


def bench_heston_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_heston_model": _bench_heston_model(seed)}
