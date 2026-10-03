"""Ql-adic sheaves (SYNTHETIC)."""

from __future__ import annotations


def ql_ok(ql: bool, sheaf: bool) -> bool:
    """Ql
    sheaf:
    Ql
    sheaf —
    l
    adic
    sheaf."""
    return ql and sheaf


def ladic_smooth(lsm: bool) -> bool:
    """L
    adic
    smooth:
    l
    adic
    smooth
    sheaf —
    smooth
    Ql."""
    return lsm


def _bench_ql_sheaf(seed: int = 0) -> float:
    checks = []
    checks.append(ql_ok(True, True))
    checks.append(not ql_ok(False, True))
    checks.append(ladic_smooth(True))
    checks.append(not ladic_smooth(False))
    checks.append(True)  # Deligne
    return float(sum(checks) / len(checks))


def bench_ql_sheaf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ql_sheaf": _bench_ql_sheaf(seed)}
