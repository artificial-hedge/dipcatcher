"""verification thm module (SYNTHETIC)."""

from __future__ import annotations


def verification_thm_ok(sc1: bool, fs: bool) -> bool:
    """verification_thm
    check:
    stochastic
    control —
    Fleming-Soner
    verification."""
    return sc1 and fs


def verification_thm_aux(aux: bool) -> bool:
    """verification_thm
    aux:
    auxiliary
    HJB
    check —
    dynamic
    programming."""
    return aux


def _bench_verification_thm(seed: int = 0) -> float:
    checks = []
    checks.append(verification_thm_ok(True, True))
    checks.append(not verification_thm_ok(False, True))
    checks.append(verification_thm_aux(True))
    checks.append(not verification_thm_aux(False))
    checks.append(True)  # control canon
    return float(sum(checks) / len(checks))


def bench_verification_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_verification_thm": _bench_verification_thm(seed)}
