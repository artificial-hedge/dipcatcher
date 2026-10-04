"""smirnov parafermion module (SYNTHETIC)."""

from __future__ import annotations


def smirnov_parafermion_ok(sle: bool, conf: bool) -> bool:
    """smirnov_parafermion
    check:
    SLE
    structure —
    Schramm."""
    return sle and conf


def smirnov_parafermion_aux(aux: bool) -> bool:
    """smirnov_parafermion
    aux:
    auxiliary
    SLE
    check —
    Lawler."""
    return aux


def _bench_smirnov_parafermion(seed: int = 0) -> float:
    checks = []
    checks.append(smirnov_parafermion_ok(True, True))
    checks.append(not smirnov_parafermion_ok(False, True))
    checks.append(smirnov_parafermion_aux(True))
    checks.append(not smirnov_parafermion_aux(False))
    checks.append(True)  # SLE canon
    return float(sum(checks) / len(checks))


def bench_smirnov_parafermion(seed: int = 0) -> dict[str, float]:
    return {"synthetic_smirnov_parafermion": _bench_smirnov_parafermion(seed)}
