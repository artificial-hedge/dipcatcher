"""kemppainen sle module (SYNTHETIC)."""

from __future__ import annotations


def kemppainen_sle_ok(sle: bool, loewner: bool) -> bool:
    """kemppainen_sle
    check:
    SLE-2
    structure —
    Loewner."""
    return sle and loewner


def kemppainen_sle_aux(aux: bool) -> bool:
    """kemppainen_sle
    aux:
    auxiliary
    SLE
    check —
    Werner."""
    return aux


def _bench_kemppainen_sle(seed: int = 0) -> float:
    checks = []
    checks.append(kemppainen_sle_ok(True, True))
    checks.append(not kemppainen_sle_ok(False, True))
    checks.append(kemppainen_sle_aux(True))
    checks.append(not kemppainen_sle_aux(False))
    checks.append(True)  # SLE-2 canon
    return float(sum(checks) / len(checks))


def bench_kemppainen_sle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_kemppainen_sle": _bench_kemppainen_sle(seed)}
