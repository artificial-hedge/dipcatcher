"""homotopy fixed module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_fixed_ok(homotopy: bool, stable: bool) -> bool:
    """homotopy_fixed
    check:
    homotopy
    structure —
    stable."""
    return homotopy and stable


def homotopy_fixed_aux(aux: bool) -> bool:
    """homotopy_fixed
    aux:
    auxiliary
    homotopy
    check —
    stable."""
    return aux


def _bench_homotopy_fixed(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_fixed_ok(True, True))
    checks.append(not homotopy_fixed_ok(False, True))
    checks.append(homotopy_fixed_aux(True))
    checks.append(not homotopy_fixed_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_homotopy_fixed(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_fixed": _bench_homotopy_fixed(seed)}
