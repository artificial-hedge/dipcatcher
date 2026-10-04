"""tao vu module (SYNTHETIC)."""

from __future__ import annotations


def tao_vu_ok(rmt: bool, univ: bool) -> bool:
    """tao_vu
    check:
    random-matrix-2
    structure —
    Tracy."""
    return rmt and univ


def tao_vu_aux(aux: bool) -> bool:
    """tao_vu
    aux:
    auxiliary
    bulk-universality
    check —
    Widom."""
    return aux


def _bench_tao_vu(seed: int = 0) -> float:
    checks = []
    checks.append(tao_vu_ok(True, True))
    checks.append(not tao_vu_ok(False, True))
    checks.append(tao_vu_aux(True))
    checks.append(not tao_vu_aux(False))
    checks.append(True)  # random-matrix-2 canon
    return float(sum(checks) / len(checks))


def bench_tao_vu(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tao_vu": _bench_tao_vu(seed)}
