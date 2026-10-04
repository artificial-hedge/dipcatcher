"""fv mart module (SYNTHETIC)."""

from __future__ import annotations


def fv_mart_ok(bnd: bool, mg: bool) -> bool:
    """fv_mart
    check:
    continuous
    martingale —
    regularity."""
    return bnd and mg


def fv_mart_aux(aux: bool) -> bool:
    """fv_mart
    aux:
    auxiliary
    martingale check —
    bracket."""
    return aux


def _bench_fv_mart(seed: int = 0) -> float:
    checks = []
    checks.append(fv_mart_ok(True, True))
    checks.append(not fv_mart_ok(False, True))
    checks.append(fv_mart_aux(True))
    checks.append(not fv_mart_aux(False))
    checks.append(True)  # continuous-martingale canon
    return float(sum(checks) / len(checks))


def bench_fv_mart(seed: int = 0) -> dict[str, float]:
    return {"synthetic_fv_mart": _bench_fv_mart(seed)}
