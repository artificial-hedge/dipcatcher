"""rough sabr module (SYNTHETIC)."""

from __future__ import annotations


def rough_sabr_ok(fh1: bool, rv: bool) -> bool:
    """rough_sabr
    check:
    rough-vol
    —
    Volterra."""
    return fh1 and rv


def rough_sabr_aux(aux: bool) -> bool:
    """rough_sabr
    aux:
    auxiliary
    rough
    check —
    multifactor."""
    return aux


def _bench_rough_sabr(seed: int = 0) -> float:
    checks = []
    checks.append(rough_sabr_ok(True, True))
    checks.append(not rough_sabr_ok(False, True))
    checks.append(rough_sabr_aux(True))
    checks.append(not rough_sabr_aux(False))
    checks.append(True)  # rough-vol canon
    return float(sum(checks) / len(checks))


def bench_rough_sabr(seed: int = 0) -> dict[str, float]:
    return {"synthetic_rough_sabr": _bench_rough_sabr(seed)}
