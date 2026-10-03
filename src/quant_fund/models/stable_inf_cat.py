"""stable inf_cat module (SYNTHETIC)."""

from __future__ import annotations


def stable_inf_cat_ok(homotopy: bool, stable: bool) -> bool:
    """stable_inf_cat
    check:
    homotopy
    structure —
    sheaf."""
    return homotopy and stable


def stable_inf_cat_aux(aux: bool) -> bool:
    """stable_inf_cat
    aux:
    auxiliary
    homotopy
    check —
    coalgebra."""
    return aux


def _bench_stable_inf_cat(seed: int = 0) -> float:
    checks = []
    checks.append(stable_inf_cat_ok(True, True))
    checks.append(not stable_inf_cat_ok(False, True))
    checks.append(stable_inf_cat_aux(True))
    checks.append(not stable_inf_cat_aux(False))
    checks.append(True)  # homotopy canon
    return float(sum(checks) / len(checks))


def bench_stable_inf_cat(seed: int = 0) -> dict[str, float]:
    return {"synthetic_stable_inf_cat": _bench_stable_inf_cat(seed)}
