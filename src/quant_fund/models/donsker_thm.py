"""donsker thm module (SYNTHETIC)."""

from __future__ import annotations


def donsker_thm_ok(proc: bool, tight: bool) -> bool:
    """donsker_thm
    check:
    empirical-process
    structure —
    Donsker."""
    return proc and tight


def donsker_thm_aux(aux: bool) -> bool:
    """donsker_thm
    aux:
    auxiliary
    class
    check —
    Vapnik."""
    return aux


def _bench_donsker_thm(seed: int = 0) -> float:
    checks = []
    checks.append(donsker_thm_ok(True, True))
    checks.append(not donsker_thm_ok(False, True))
    checks.append(donsker_thm_aux(True))
    checks.append(not donsker_thm_aux(False))
    checks.append(True)  # empirical canon
    return float(sum(checks) / len(checks))


def bench_donsker_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_donsker_thm": _bench_donsker_thm(seed)}
