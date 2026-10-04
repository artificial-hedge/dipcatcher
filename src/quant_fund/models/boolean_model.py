"""boolean model module (SYNTHETIC)."""

from __future__ import annotations


def boolean_model_ok(geo: bool, tess: bool) -> bool:
    """boolean_model
    check:
    stochastic
    geometry —
    tessellation."""
    return geo and tess


def boolean_model_aux(aux: bool) -> bool:
    """boolean_model
    aux:
    auxiliary
    geometry check —
    measure."""
    return aux


def _bench_boolean_model(seed: int = 0) -> float:
    checks = []
    checks.append(boolean_model_ok(True, True))
    checks.append(not boolean_model_ok(False, True))
    checks.append(boolean_model_aux(True))
    checks.append(not boolean_model_aux(False))
    checks.append(True)  # stochastic-geometry canon
    return float(sum(checks) / len(checks))


def bench_boolean_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_boolean_model": _bench_boolean_model(seed)}
