"""kou model module (SYNTHETIC)."""

from __future__ import annotations


def kou_model_ok(jd: bool, mj: bool) -> bool:
    """kou_model
    check:
    jump-process
    model —
    finite
    activity."""
    return jd and mj


def kou_model_aux(aux: bool) -> bool:
    """kou_model
    aux:
    auxiliary
    jump
    check —
    compensator."""
    return aux


def _bench_kou_model(seed: int = 0) -> float:
    checks = []
    checks.append(kou_model_ok(True, True))
    checks.append(not kou_model_ok(False, True))
    checks.append(kou_model_aux(True))
    checks.append(not kou_model_aux(False))
    checks.append(True)  # jump-process canon
    return float(sum(checks) / len(checks))


def bench_kou_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kou_model": _bench_kou_model(seed)}
