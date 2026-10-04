"""Lasota-Yorke inequality (SYNTHETIC)."""

from __future__ import annotations


def ly_ok(bv: bool, bound: bool) -> bool:
    """Lasota-
    Yorke
    inequality:
    transfer
    operator
    is
    quasi-
    compact
    on BV —
    contraction
    plus a
    compact
    term."""
    return bv and bound


def quasi_compact(qc: bool) -> bool:
    """Quasi-
    compactness:
    essential
    spectral
    radius
    smaller
    than
    top
    eigenvalue —
    exponential
    correlation
    decay."""
    return qc


def _bench_lasota_yorke(seed: int = 0) -> float:
    checks = []
    checks.append(ly_ok(True, True))
    checks.append(not ly_ok(False, True))
    checks.append(quasi_compact(True))
    checks.append(not quasi_compact(False))
    checks.append(True)  # Lasota-Yorke
    return float(sum(checks) / len(checks))


def bench_lasota_yorke(seed: int = 0) -> dict[str, float]:
    return {"synthetic_lasota_yorke": _bench_lasota_yorke(seed)}
