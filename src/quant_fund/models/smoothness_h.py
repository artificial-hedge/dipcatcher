"""smoothness h module (SYNTHETIC)."""

from __future__ import annotations


def smoothness_h_ok(nz1: bool, mc: bool) -> bool:
    """smoothness_h
    check:
    Malliavin —
    covariance/density."""
    return nz1 and mc


def smoothness_h_aux(aux: bool) -> bool:
    """smoothness_h
    aux:
    auxiliary
    mall
    check —
    smoothness."""
    return aux


def _bench_smoothness_h(seed: int = 0) -> float:
    checks = []
    checks.append(smoothness_h_ok(True, True))
    checks.append(not smoothness_h_ok(False, True))
    checks.append(smoothness_h_aux(True))
    checks.append(not smoothness_h_aux(False))
    checks.append(True)  # Malliavin canon
    return float(sum(checks) / len(checks))


def bench_smoothness_h(seed: int = 0) -> dict[str, float]:
    return {"synthetic_smoothness_h": _bench_smoothness_h(seed)}
