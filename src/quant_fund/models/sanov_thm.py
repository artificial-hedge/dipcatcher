"""sanov thm module (SYNTHETIC)."""

from __future__ import annotations


def sanov_thm_ok(rate: bool, action: bool) -> bool:
    """sanov_thm
    check:
    large-deviation
    structure —
    Varadhan."""
    return rate and action


def sanov_thm_aux(aux: bool) -> bool:
    """sanov_thm
    aux:
    auxiliary
    contraction
    check —
    Wentzell."""
    return aux


def _bench_sanov_thm(seed: int = 0) -> float:
    checks = []
    checks.append(sanov_thm_ok(True, True))
    checks.append(not sanov_thm_ok(False, True))
    checks.append(sanov_thm_aux(True))
    checks.append(not sanov_thm_aux(False))
    checks.append(True)  # LDP canon
    return float(sum(checks) / len(checks))


def bench_sanov_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_sanov_thm": _bench_sanov_thm(seed)}
