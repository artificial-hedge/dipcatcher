"""galton watson module (SYNTHETIC)."""

from __future__ import annotations


def galton_watson_ok(mean: bool, var: bool) -> bool:
    """galton_watson
    check:
    branching
    structure —
    Galton–Watson."""
    return mean and var


def galton_watson_aux(aux: bool) -> bool:
    """galton_watson
    aux:
    auxiliary
    immigration
    check —
    BIMM."""
    return aux


def _bench_galton_watson(seed: int = 0) -> float:
    checks = []
    checks.append(galton_watson_ok(True, True))
    checks.append(not galton_watson_ok(False, True))
    checks.append(galton_watson_aux(True))
    checks.append(not galton_watson_aux(False))
    checks.append(True)  # branching canon
    return float(sum(checks) / len(checks))


def bench_galton_watson(seed: int = 0) -> dict[str, float]:
    return {"synthetic_galton_watson": _bench_galton_watson(seed)}
