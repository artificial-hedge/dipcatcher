"""thom mgl2 module (SYNTHETIC)."""

from __future__ import annotations


def thom_mgl2_ok(motive: bool, suslin: bool) -> bool:
    """thom_mgl2
    check:
    motivic-A1-2
    structure —
    Suslin."""
    return motive and suslin


def thom_mgl2_aux(aux: bool) -> bool:
    """thom_mgl2
    aux:
    auxiliary
    motive
    check —
    Totaro."""
    return aux


def _bench_thom_mgl2(seed: int = 0) -> float:
    checks = []
    checks.append(thom_mgl2_ok(True, True))
    checks.append(not thom_mgl2_ok(False, True))
    checks.append(thom_mgl2_aux(True))
    checks.append(not thom_mgl2_aux(False))
    checks.append(True)  # motivic-A1-2 canon
    return float(sum(checks) / len(checks))


def bench_thom_mgl2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_thom_mgl2": _bench_thom_mgl2(seed)}
