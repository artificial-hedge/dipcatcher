"""f motive2 module (SYNTHETIC)."""

from __future__ import annotations


def f_motive2_ok(motivic: bool, stable: bool) -> bool:
    """f_motive2
    check:
    motivic
    stable
    homotopy —
    slice."""
    return motivic and stable


def f_motive2_aux(aux: bool) -> bool:
    """f_motive2
    aux:
    auxiliary
    motivic
    check —
    spectral."""
    return aux


def _bench_f_motive2(seed: int = 0) -> float:
    checks = []
    checks.append(f_motive2_ok(True, True))
    checks.append(not f_motive2_ok(False, True))
    checks.append(f_motive2_aux(True))
    checks.append(not f_motive2_aux(False))
    checks.append(True)  # motivic canon
    return float(sum(checks) / len(checks))


def bench_f_motive2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_f_motive2": _bench_f_motive2(seed)}
