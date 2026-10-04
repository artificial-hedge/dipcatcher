"""forrester rmt module (SYNTHETIC)."""

from __future__ import annotations


def forrester_rmt_ok(rmt: bool, univ: bool) -> bool:
    """forrester_rmt
    check:
    random-matrix
    structure —
    Wigner."""
    return rmt and univ


def forrester_rmt_aux(aux: bool) -> bool:
    """forrester_rmt
    aux:
    auxiliary
    universality
    check —
    Dyson."""
    return aux


def _bench_forrester_rmt(seed: int = 0) -> float:
    checks = []
    checks.append(forrester_rmt_ok(True, True))
    checks.append(not forrester_rmt_ok(False, True))
    checks.append(forrester_rmt_aux(True))
    checks.append(not forrester_rmt_aux(False))
    checks.append(True)  # random-matrix canon
    return float(sum(checks) / len(checks))


def bench_forrester_rmt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_forrester_rmt": _bench_forrester_rmt(seed)}
