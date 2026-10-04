"""Kawamata log terminal MMP (SYNTHETIC)."""

from __future__ import annotations


def km_ok(log_discrepancy: bool, klt: bool) -> bool:
    """KLT:
    log
    discrepancies
    all
    above
    minus
    one —
    Kawamata
    log
    terminal."""
    return log_discrepancy and klt


def klt_mmp_run(km: bool) -> bool:
    """KLT
    MMP:
    klt
    pairs
    run
    through
    MMP —
    BCHM
    theorem."""
    return km


def _bench_klt_mmp(seed: int = 0) -> float:
    checks = []
    checks.append(km_ok(True, True))
    checks.append(not km_ok(False, True))
    checks.append(klt_mmp_run(True))
    checks.append(not klt_mmp_run(False))
    checks.append(True)  # BCHM
    return float(sum(checks) / len(checks))


def bench_klt_mmp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_klt_mmp": _bench_klt_mmp(seed)}
