"""benoist sle module (SYNTHETIC)."""

from __future__ import annotations


def benoist_sle_ok(sle: bool, loewner: bool) -> bool:
    """benoist_sle
    check:
    SLE-2
    structure —
    Loewner."""
    return sle and loewner


def benoist_sle_aux(aux: bool) -> bool:
    """benoist_sle
    aux:
    auxiliary
    SLE
    check —
    Werner."""
    return aux


def _bench_benoist_sle(seed: int = 0) -> float:
    checks = []
    checks.append(benoist_sle_ok(True, True))
    checks.append(not benoist_sle_ok(False, True))
    checks.append(benoist_sle_aux(True))
    checks.append(not benoist_sle_aux(False))
    checks.append(True)  # SLE-2 canon
    return float(sum(checks) / len(checks))


def bench_benoist_sle(seed: int = 0) -> dict[str, float]:
    return {"synthetic_benoist_sle": _bench_benoist_sle(seed)}
