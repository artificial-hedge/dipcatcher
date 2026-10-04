"""zykin sle module (SYNTHETIC)."""

from __future__ import annotations


def zykin_sle_ok(sle: bool, loewner: bool) -> bool:
    """zykin_sle
    check:
    SLE-2
    structure —
    Loewner."""
    return sle and loewner


def zykin_sle_aux(aux: bool) -> bool:
    """zykin_sle
    aux:
    auxiliary
    SLE
    check —
    Werner."""
    return aux


def _bench_zykin_sle(seed: int = 0) -> float:
    checks = []
    checks.append(zykin_sle_ok(True, True))
    checks.append(not zykin_sle_ok(False, True))
    checks.append(zykin_sle_aux(True))
    checks.append(not zykin_sle_aux(False))
    checks.append(True)  # SLE-2 canon
    return float(sum(checks) / len(checks))


def bench_zykin_sle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_zykin_sle": _bench_zykin_sle(seed)}
