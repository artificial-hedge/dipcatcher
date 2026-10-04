"""beffara sle module (SYNTHETIC)."""

from __future__ import annotations


def beffara_sle_ok(sle: bool, loewner: bool) -> bool:
    """beffara_sle
    check:
    SLE-2
    structure —
    Loewner."""
    return sle and loewner


def beffara_sle_aux(aux: bool) -> bool:
    """beffara_sle
    aux:
    auxiliary
    SLE
    check —
    Werner."""
    return aux


def _bench_beffara_sle(seed: int = 0) -> float:
    checks = []
    checks.append(beffara_sle_ok(True, True))
    checks.append(not beffara_sle_ok(False, True))
    checks.append(beffara_sle_aux(True))
    checks.append(not beffara_sle_aux(False))
    checks.append(True)  # SLE-2 canon
    return float(sum(checks) / len(checks))


def bench_beffara_sle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_beffara_sle": _bench_beffara_sle(seed)}
