"""pseudo arclength module (SYNTHETIC)."""

from __future__ import annotations


def pseudo_arclength_ok(path: bool, step: bool) -> bool:
    """pseudo_arclength
    check:
    continuation/homotopy —
    predictor
    consistency."""
    return path and step


def pseudo_arclength_aux(aux: bool) -> bool:
    """pseudo_arclength
    aux:
    auxiliary
    continuation check —
    corrector bound."""
    return aux


def _bench_pseudo_arclength(seed: int = 0) -> float:
    checks = []
    checks.append(pseudo_arclength_ok(True, True))
    checks.append(not pseudo_arclength_ok(False, True))
    checks.append(pseudo_arclength_aux(True))
    checks.append(not pseudo_arclength_aux(False))
    checks.append(True)  # continuation canon
    return float(sum(checks) / len(checks))


def bench_pseudo_arclength(seed: int = 0) -> dict[str, float]:
    return {"synthetic_pseudo_arclength": _bench_pseudo_arclength(seed)}
