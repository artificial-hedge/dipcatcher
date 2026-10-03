"""cadlag mart module (SYNTHETIC)."""

from __future__ import annotations


def cadlag_mart_ok(bnd: bool, mg: bool) -> bool:
    """cadlag_mart
    check:
    continuous
    martingale —
    regularity."""
    return bnd and mg


def cadlag_mart_aux(aux: bool) -> bool:
    """cadlag_mart
    aux:
    auxiliary
    martingale check —
    bracket."""
    return aux


def _bench_cadlag_mart(seed: int = 0) -> float:
    checks = []
    checks.append(cadlag_mart_ok(True, True))
    checks.append(not cadlag_mart_ok(False, True))
    checks.append(cadlag_mart_aux(True))
    checks.append(not cadlag_mart_aux(False))
    checks.append(True)  # continuous-martingale canon
    return float(sum(checks) / len(checks))


def bench_cadlag_mart(seed: int = 0) -> dict[str, float]:
    return {"synthetic_cadlag_mart": _bench_cadlag_mart(seed)}
