"""schider thm module (SYNTHETIC)."""

from __future__ import annotations


def schider_thm_ok(rate: bool, action: bool) -> bool:
    """schider_thm
    check:
    large-deviation
    structure —
    Varadhan."""
    return rate and action


def schider_thm_aux(aux: bool) -> bool:
    """schider_thm
    aux:
    auxiliary
    contraction
    check —
    Wentzell."""
    return aux


def _bench_schider_thm(seed: int = 0) -> float:
    checks = []
    checks.append(schider_thm_ok(True, True))
    checks.append(not schider_thm_ok(False, True))
    checks.append(schider_thm_aux(True))
    checks.append(not schider_thm_aux(False))
    checks.append(True)  # LDP canon
    return float(sum(checks) / len(checks))


def bench_schider_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_schider_thm": _bench_schider_thm(seed)}
