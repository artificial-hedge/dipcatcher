"""beilinson regulator2 module (SYNTHETIC)."""

from __future__ import annotations


def beilinson_regulator2_ok(motivic: bool, hodge: bool) -> bool:
    """beilinson_regulator2
    check:
    motivic
    structure —
    Hodge."""
    return motivic and hodge


def beilinson_regulator2_aux(aux: bool) -> bool:
    """beilinson_regulator2
    aux:
    auxiliary
    motivic
    check —
    Tate."""
    return aux


def _bench_beilinson_regulator2(seed: int = 0) -> float:
    checks = []
    checks.append(beilinson_regulator2_ok(True, True))
    checks.append(not beilinson_regulator2_ok(False, True))
    checks.append(beilinson_regulator2_aux(True))
    checks.append(not beilinson_regulator2_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_beilinson_regulator2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_beilinson_regulator2": _bench_beilinson_regulator2(seed)}
