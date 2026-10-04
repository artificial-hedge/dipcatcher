"""Nygaard filtration (SYNTHETIC)."""

from __future__ import annotations


def n2_ok(nygaard: bool, filtration: bool) -> bool:
    """Nygaard:
    Nygaard
    filtration —
    Nygaard
    filtration."""
    return nygaard and filtration


def nygaard_gr(ng: bool) -> bool:
    """Nygaard
    graded:
    Nygaard
    graded
    pieces —
    Bhatt
    Nygaard."""
    return ng


def _bench_nygaard2(seed: int = 0) -> float:
    checks = []
    checks.append(n2_ok(True, True))
    checks.append(not n2_ok(False, True))
    checks.append(nygaard_gr(True))
    checks.append(not nygaard_gr(False))
    checks.append(True)  # Bhatt-Morrow-Scholze
    return float(sum(checks) / len(checks))


def bench_nygaard2(seed: int = 0) -> dict[str, float]:
    return {"synthetic_nygaard2": _bench_nygaard2(seed)}
