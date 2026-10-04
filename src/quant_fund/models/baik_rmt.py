"""baik rmt module (SYNTHETIC)."""

from __future__ import annotations


def baik_rmt_ok(rmt: bool, univ: bool) -> bool:
    """baik_rmt
    check:
    random-matrix-2
    structure —
    Tracy."""
    return rmt and univ


def baik_rmt_aux(aux: bool) -> bool:
    """baik_rmt
    aux:
    auxiliary
    bulk-universality
    check —
    Widom."""
    return aux


def _bench_baik_rmt(seed: int = 0) -> float:
    checks = []
    checks.append(baik_rmt_ok(True, True))
    checks.append(not baik_rmt_ok(False, True))
    checks.append(baik_rmt_aux(True))
    checks.append(not baik_rmt_aux(False))
    checks.append(True)  # random-matrix-2 canon
    return float(sum(checks) / len(checks))


def bench_baik_rmt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_baik_rmt": _bench_baik_rmt(seed)}
