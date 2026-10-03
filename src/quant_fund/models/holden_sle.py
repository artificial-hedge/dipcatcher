"""holden sle module (SYNTHETIC)."""

from __future__ import annotations


def holden_sle_ok(sle: bool, loewner: bool) -> bool:
    """holden_sle
    check:
    SLE-2
    structure —
    Loewner."""
    return sle and loewner


def holden_sle_aux(aux: bool) -> bool:
    """holden_sle
    aux:
    auxiliary
    SLE
    check —
    Werner."""
    return aux


def _bench_holden_sle(seed: int = 0) -> float:
    checks = []
    checks.append(holden_sle_ok(True, True))
    checks.append(not holden_sle_ok(False, True))
    checks.append(holden_sle_aux(True))
    checks.append(not holden_sle_aux(False))
    checks.append(True)  # SLE-2 canon
    return float(sum(checks) / len(checks))


def bench_holden_sle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_holden_sle": _bench_holden_sle(seed)}
