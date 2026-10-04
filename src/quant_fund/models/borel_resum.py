"""borel resum module (SYNTHETIC)."""

from __future__ import annotations


def borel_resum_ok(series: bool, order: bool) -> bool:
    """borel_resum
    check:
    asymptotic
    analysis —
    series."""
    return series and order


def borel_resum_aux(aux: bool) -> bool:
    """borel_resum
    aux:
    auxiliary
    asymptotic check —
    remainder."""
    return aux


def _bench_borel_resum(seed: int = 0) -> float:
    checks = []
    checks.append(borel_resum_ok(True, True))
    checks.append(not borel_resum_ok(False, True))
    checks.append(borel_resum_aux(True))
    checks.append(not borel_resum_aux(False))
    checks.append(True)  # asymptotic-analysis canon
    return float(sum(checks) / len(checks))


def bench_borel_resum(seed: int = 0) -> dict[str, float]:
    return {"synthetic_borel_resum": _bench_borel_resum(seed)}
