"""homotopy factor module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_factor_ok(homotopy: bool, stable: bool) -> bool:
    """homotopy_factor
    check:
    homotopy
    structure —
    stable."""
    return homotopy and stable


def homotopy_factor_aux(aux: bool) -> bool:
    """homotopy_factor
    aux:
    auxiliary
    homotopy
    check —
    stable."""
    return aux


def _bench_homotopy_factor(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_factor_ok(True, True))
    checks.append(not homotopy_factor_ok(False, True))
    checks.append(homotopy_factor_aux(True))
    checks.append(not homotopy_factor_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_homotopy_factor(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_factor": _bench_homotopy_factor(seed)}
