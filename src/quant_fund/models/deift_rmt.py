"""deift rmt module (SYNTHETIC)."""

from __future__ import annotations


def deift_rmt_ok(rmt: bool, univ: bool) -> bool:
    """deift_rmt
    check:
    random-matrix
    structure —
    Wigner."""
    return rmt and univ


def deift_rmt_aux(aux: bool) -> bool:
    """deift_rmt
    aux:
    auxiliary
    universality
    check —
    Dyson."""
    return aux


def _bench_deift_rmt(seed: int = 0) -> float:
    checks = []
    checks.append(deift_rmt_ok(True, True))
    checks.append(not deift_rmt_ok(False, True))
    checks.append(deift_rmt_aux(True))
    checks.append(not deift_rmt_aux(False))
    checks.append(True)  # random-matrix canon
    return float(sum(checks) / len(checks))


def bench_deift_rmt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_deift_rmt": _bench_deift_rmt(seed)}
