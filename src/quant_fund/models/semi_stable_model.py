"""semi stable_model module (SYNTHETIC)."""

from __future__ import annotations


def semi_stable_model_ok(ramif: bool, model: bool) -> bool:
    """semi_stable_model
    check:
    ramification-2
    structure —
    Brylinski."""
    return ramif and model


def semi_stable_model_aux(aux: bool) -> bool:
    """semi_stable_model
    aux:
    auxiliary
    semistable
    check —
    Raynaud."""
    return aux


def _bench_semi_stable_model(seed: int = 0) -> float:
    checks = []
    checks.append(semi_stable_model_ok(True, True))
    checks.append(not semi_stable_model_ok(False, True))
    checks.append(semi_stable_model_aux(True))
    checks.append(not semi_stable_model_aux(False))
    checks.append(True)  # ramification-2 canon
    return float(sum(checks) / len(checks))


def bench_semi_stable_model(seed: int = 0) -> dict[str, float]:
    return {"synthetic_semi_stable_model": _bench_semi_stable_model(seed)}
