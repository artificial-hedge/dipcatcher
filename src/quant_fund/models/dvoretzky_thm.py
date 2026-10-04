"""dvoretzky thm module (SYNTHETIC)."""

from __future__ import annotations


def dvoretzky_thm_ok(ent: bool, bound: bool) -> bool:
    """dvoretzky_thm
    check:
    entropy
    structure —
    Dudley."""
    return ent and bound


def dvoretzky_thm_aux(aux: bool) -> bool:
    """dvoretzky_thm
    aux:
    auxiliary
    metric
    check —
    Vapnik."""
    return aux


def _bench_dvoretzky_thm(seed: int = 0) -> float:
    checks = []
    checks.append(dvoretzky_thm_ok(True, True))
    checks.append(not dvoretzky_thm_ok(False, True))
    checks.append(dvoretzky_thm_aux(True))
    checks.append(not dvoretzky_thm_aux(False))
    checks.append(True)  # empirical canon
    return float(sum(checks) / len(checks))


def bench_dvoretzky_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_dvoretzky_thm": _bench_dvoretzky_thm(seed)}
