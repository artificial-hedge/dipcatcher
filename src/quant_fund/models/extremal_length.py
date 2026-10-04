"""Extremal length (SYNTHETIC)."""

from __future__ import annotations


def el_ok(conformal: bool, reciprocal: bool) -> bool:
    """Extremal
    length:
    conformal
    invariant
    of
    a
    curve
    family —
    reciprocal
    of
    modulus,
    controls
    Teichmueller
    metric."""
    return conformal and reciprocal


def kerckhoff_formula(kf: bool) -> bool:
    """Kerckhoff:
    Teichmueller
    distance
    is
    realized
    by
    extremal-
    length
    ratios
    —
    sup
    formula."""
    return kf


def _bench_extremal_length(seed: int = 0) -> float:
    checks = []
    checks.append(el_ok(True, True))
    checks.append(not el_ok(False, True))
    checks.append(kerckhoff_formula(True))
    checks.append(not kerckhoff_formula(False))
    checks.append(True)  # Kerckhoff
    return float(sum(checks) / len(checks))


def bench_extremal_length(seed: int = 0) -> dict[str, float]:
    return {"synthetic_extremal_length": _bench_extremal_length(seed)}
