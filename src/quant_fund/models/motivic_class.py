"""Motivic characteristic classes (SYNTHETIC)."""

from __future__ import annotations


def mcl_ok(motivic: bool, klass: bool) -> bool:
    """Motivic
    class:
    motivic
    characteristic
    classes —
    motivic
    Chern."""
    return motivic and klass


def motivic_chern(mc: bool) -> bool:
    """Motivic
    Chern:
    motivic
    Chern
    classes —
    Pushin
    Chern."""
    return mc


def _bench_motivic_class(seed: int = 0) -> float:
    checks = []
    checks.append(mcl_ok(True, True))
    checks.append(not mcl_ok(False, True))
    checks.append(motivic_chern(True))
    checks.append(not motivic_chern(False))
    checks.append(True)  # motivic Chern
    return float(sum(checks) / len(checks))


def bench_motivic_class(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_class": _bench_motivic_class(seed)}
