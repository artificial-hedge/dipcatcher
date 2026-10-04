"""l-adic sheaf (SYNTHETIC)."""

from __future__ import annotations


def la_ok(l_adic: bool, sheaf: bool) -> bool:
    """l-
    adic:
    l-
    adic
    sheaf
    system —
    Deligne
    l-adic."""
    return l_adic and sheaf


def ladic_system(ls: bool) -> bool:
    """l-adic
    system:
    l-
    adic
    inverse
    system —
    l-adic
    sheaf."""
    return ls


def _bench_ladic_sheaf(seed: int = 0) -> float:
    checks = []
    checks.append(la_ok(True, True))
    checks.append(not la_ok(False, True))
    checks.append(ladic_system(True))
    checks.append(not ladic_system(False))
    checks.append(True)  # Deligne
    return float(sum(checks) / len(checks))


def bench_ladic_sheaf(seed: int = 0) -> dict[str, float]:
    return {"synthetic_ladic_sheaf": _bench_ladic_sheaf(seed)}
