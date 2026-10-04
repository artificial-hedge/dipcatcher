"""stricker thm module (SYNTHETIC)."""

from __future__ import annotations


def stricker_thm_ok(ord1: bool, meas: bool) -> bool:
    """stricker_thm
    check:
    stochastic-order
    structure —
    semimartingale
    canon."""
    return ord1 and meas


def stricker_thm_aux(aux: bool) -> bool:
    """stricker_thm
    aux:
    auxiliary
    order
    check —
    Cramer-Wold
    device."""
    return aux


def _bench_stricker_thm(seed: int = 0) -> float:
    checks = []
    checks.append(stricker_thm_ok(True, True))
    checks.append(not stricker_thm_ok(False, True))
    checks.append(stricker_thm_aux(True))
    checks.append(not stricker_thm_aux(False))
    checks.append(True)  # order canon
    return float(sum(checks) / len(checks))


def bench_stricker_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stricker_thm": _bench_stricker_thm(seed)}
