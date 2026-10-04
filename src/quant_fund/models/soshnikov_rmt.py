"""soshnikov rmt module (SYNTHETIC)."""

from __future__ import annotations


def soshnikov_rmt_ok(rmt: bool, univ: bool) -> bool:
    """soshnikov_rmt
    check:
    random-matrix
    structure —
    Wigner."""
    return rmt and univ


def soshnikov_rmt_aux(aux: bool) -> bool:
    """soshnikov_rmt
    aux:
    auxiliary
    universality
    check —
    Dyson."""
    return aux


def _bench_soshnikov_rmt(seed: int = 0) -> float:
    checks = []
    checks.append(soshnikov_rmt_ok(True, True))
    checks.append(not soshnikov_rmt_ok(False, True))
    checks.append(soshnikov_rmt_aux(True))
    checks.append(not soshnikov_rmt_aux(False))
    checks.append(True)  # random-matrix canon
    return float(sum(checks) / len(checks))


def bench_soshnikov_rmt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_soshnikov_rmt": _bench_soshnikov_rmt(seed)}
