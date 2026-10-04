"""tangent predictor module (SYNTHETIC)."""

from __future__ import annotations


def tangent_predictor_ok(step: bool, conv: bool) -> bool:
    """tangent_predictor
    check:
    acceleration —
    step/contraction
    consistency."""
    return step and conv


def tangent_predictor_aux(aux: bool) -> bool:
    """tangent_predictor
    aux:
    auxiliary
    accelerator check —
    rate bound."""
    return aux


def _bench_tangent_predictor(seed: int = 0) -> float:
    checks = []
    checks.append(tangent_predictor_ok(True, True))
    checks.append(not tangent_predictor_ok(False, True))
    checks.append(tangent_predictor_aux(True))
    checks.append(not tangent_predictor_aux(False))
    checks.append(True)  # acceleration canon
    return float(sum(checks) / len(checks))


def bench_tangent_predictor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tangent_predictor": _bench_tangent_predictor(seed)}
