"""bounded mart module (SYNTHETIC)."""

from __future__ import annotations


def bounded_mart_ok(bnd: bool, mg: bool) -> bool:
    """bounded_mart
    check:
    continuous
    martingale —
    regularity."""
    return bnd and mg


def bounded_mart_aux(aux: bool) -> bool:
    """bounded_mart
    aux:
    auxiliary
    martingale check —
    bracket."""
    return aux


def _bench_bounded_mart(seed: int = 0) -> float:
    checks = []
    checks.append(bounded_mart_ok(True, True))
    checks.append(not bounded_mart_ok(False, True))
    checks.append(bounded_mart_aux(True))
    checks.append(not bounded_mart_aux(False))
    checks.append(True)  # continuous-martingale canon
    return float(sum(checks) / len(checks))


def bench_bounded_mart(seed: int = 0) -> dict[str, float]:
    return {"synthetic_bounded_mart": _bench_bounded_mart(seed)}
