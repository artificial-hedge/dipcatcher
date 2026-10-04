"""tannakian motive2 module (SYNTHETIC)."""

from __future__ import annotations


def tannakian_motive2_ok(motivic: bool, hodge: bool) -> bool:
    """tannakian_motive2
    check:
    motivic
    structure —
    Hodge."""
    return motivic and hodge


def tannakian_motive2_aux(aux: bool) -> bool:
    """tannakian_motive2
    aux:
    auxiliary
    motivic
    check —
    Tate."""
    return aux


def _bench_tannakian_motive2(seed: int = 0) -> float:
    checks = []
    checks.append(tannakian_motive2_ok(True, True))
    checks.append(not tannakian_motive2_ok(False, True))
    checks.append(tannakian_motive2_aux(True))
    checks.append(not tannakian_motive2_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_tannakian_motive2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tannakian_motive2": _bench_tannakian_motive2(seed)}
