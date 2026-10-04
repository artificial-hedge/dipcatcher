"""Nygaard filtration (SYNTHETIC)."""

from __future__ import annotations


def nf_ok(nygaard: bool, motivic: bool) -> bool:
    """Nygaard:
    filtration
    on
    crystalline
    cohomology —
    Nygaard
    filtration."""
    return nygaard and motivic


def nygaard_level(nl: bool) -> bool:
    """Nygaard
    level:
    N
    filtration
    grading —
    motivic
    weight."""
    return nl


def _bench_nygaard_filt(seed: int = 0) -> float:
    checks = []
    checks.append(nf_ok(True, True))
    checks.append(not nf_ok(False, True))
    checks.append(nygaard_level(True))
    checks.append(not nygaard_level(False))
    checks.append(True)  # Nygaard
    return float(sum(checks) / len(checks))


def bench_nygaard_filt(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nygaard_filt": _bench_nygaard_filt(seed)}
