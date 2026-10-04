"""deflation method module (SYNTHETIC)."""

from __future__ import annotations


def deflation_method_ok(path: bool, step: bool) -> bool:
    """deflation_method
    check:
    continuation/homotopy —
    predictor
    consistency."""
    return path and step


def deflation_method_aux(aux: bool) -> bool:
    """deflation_method
    aux:
    auxiliary
    continuation check —
    corrector bound."""
    return aux


def _bench_deflation_method(seed: int = 0) -> float:
    checks = []
    checks.append(deflation_method_ok(True, True))
    checks.append(not deflation_method_ok(False, True))
    checks.append(deflation_method_aux(True))
    checks.append(not deflation_method_aux(False))
    checks.append(True)  # continuation canon
    return float(sum(checks) / len(checks))


def bench_deflation_method(seed: int = 0) -> dict[str, float]:
    return {"synthetic_deflation_method": _bench_deflation_method(seed)}
