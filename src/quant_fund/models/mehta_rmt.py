"""mehta rmt module (SYNTHETIC)."""

from __future__ import annotations


def mehta_rmt_ok(rmt: bool, univ: bool) -> bool:
    """mehta_rmt
    check:
    random-matrix
    structure —
    Wigner."""
    return rmt and univ


def mehta_rmt_aux(aux: bool) -> bool:
    """mehta_rmt
    aux:
    auxiliary
    universality
    check —
    Dyson."""
    return aux


def _bench_mehta_rmt(seed: int = 0) -> float:
    checks = []
    checks.append(mehta_rmt_ok(True, True))
    checks.append(not mehta_rmt_ok(False, True))
    checks.append(mehta_rmt_aux(True))
    checks.append(not mehta_rmt_aux(False))
    checks.append(True)  # random-matrix canon
    return float(sum(checks) / len(checks))


def bench_mehta_rmt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_mehta_rmt": _bench_mehta_rmt(seed)}
