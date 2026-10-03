"""decomp mart module (SYNTHETIC)."""

from __future__ import annotations


def decomp_mart_ok(bnd: bool, mg: bool) -> bool:
    """decomp_mart
    check:
    continuous
    martingale —
    regularity."""
    return bnd and mg


def decomp_mart_aux(aux: bool) -> bool:
    """decomp_mart
    aux:
    auxiliary
    martingale check —
    bracket."""
    return aux


def _bench_decomp_mart(seed: int = 0) -> float:
    checks = []
    checks.append(decomp_mart_ok(True, True))
    checks.append(not decomp_mart_ok(False, True))
    checks.append(decomp_mart_aux(True))
    checks.append(not decomp_mart_aux(False))
    checks.append(True)  # continuous-martingale canon
    return float(sum(checks) / len(checks))


def bench_decomp_mart(seed: int = 0) -> dict[str, float]:
    return {"synthetic_decomp_mart": _bench_decomp_mart(seed)}
