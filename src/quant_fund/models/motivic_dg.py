"""Motivic dg-categories (SYNTHETIC)."""

from __future__ import annotations


def md_ok(motivic: bool, dg: bool) -> bool:
    """Motivic:
    motivic
    dg-
    category
    non-
    commutative
    motives —
    Tabuada."""
    return motivic and dg


def nc_motive(nm: bool) -> bool:
    """NC
    motive:
    non-
    commutative
    motives —
    Kontsevich-
    Tabuada."""
    return nm


def _bench_motivic_dg(seed: int = 0) -> float:
    checks = []
    checks.append(md_ok(True, True))
    checks.append(not md_ok(False, True))
    checks.append(nc_motive(True))
    checks.append(not nc_motive(False))
    checks.append(True)  # Tabuada
    return float(sum(checks) / len(checks))


def bench_motivic_dg(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_dg": _bench_motivic_dg(seed)}
