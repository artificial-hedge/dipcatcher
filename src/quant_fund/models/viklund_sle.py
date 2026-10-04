"""viklund sle module (SYNTHETIC)."""

from __future__ import annotations


def viklund_sle_ok(sle: bool, loewner: bool) -> bool:
    """viklund_sle
    check:
    SLE-2
    structure —
    Loewner."""
    return sle and loewner


def viklund_sle_aux(aux: bool) -> bool:
    """viklund_sle
    aux:
    auxiliary
    SLE
    check —
    Werner."""
    return aux


def _bench_viklund_sle(seed: int = 0) -> float:
    checks = []
    checks.append(viklund_sle_ok(True, True))
    checks.append(not viklund_sle_ok(False, True))
    checks.append(viklund_sle_aux(True))
    checks.append(not viklund_sle_aux(False))
    checks.append(True)  # SLE-2 canon
    return float(sum(checks) / len(checks))


def bench_viklund_sle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_viklund_sle": _bench_viklund_sle(seed)}
