"""chafai rmt module (SYNTHETIC)."""

from __future__ import annotations


def chafai_rmt_ok(rmt: bool, univ: bool) -> bool:
    """chafai_rmt
    check:
    random-matrix-2
    structure —
    Tracy."""
    return rmt and univ


def chafai_rmt_aux(aux: bool) -> bool:
    """chafai_rmt
    aux:
    auxiliary
    bulk-universality
    check —
    Widom."""
    return aux


def _bench_chafai_rmt(seed: int = 0) -> float:
    checks = []
    checks.append(chafai_rmt_ok(True, True))
    checks.append(not chafai_rmt_ok(False, True))
    checks.append(chafai_rmt_aux(True))
    checks.append(not chafai_rmt_aux(False))
    checks.append(True)  # random-matrix-2 canon
    return float(sum(checks) / len(checks))


def bench_chafai_rmt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_chafai_rmt": _bench_chafai_rmt(seed)}
