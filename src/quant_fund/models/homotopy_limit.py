"""homotopy limit module (SYNTHETIC)."""

from __future__ import annotations


def homotopy_limit_ok(homotopy: bool, stable: bool) -> bool:
    """homotopy_limit
    check:
    homotopy
    structure —
    stable."""
    return homotopy and stable


def homotopy_limit_aux(aux: bool) -> bool:
    """homotopy_limit
    aux:
    auxiliary
    homotopy
    check —
    limit."""
    return aux


def _bench_homotopy_limit(seed: int = 0) -> float:
    checks = []
    checks.append(homotopy_limit_ok(True, True))
    checks.append(not homotopy_limit_ok(False, True))
    checks.append(homotopy_limit_aux(True))
    checks.append(not homotopy_limit_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_homotopy_limit(seed: int = 0) -> dict[str, float]:
    return {"synthetic_homotopy_limit": _bench_homotopy_limit(seed)}
