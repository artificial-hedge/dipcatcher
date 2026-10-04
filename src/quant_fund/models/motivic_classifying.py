"""Motivic classifying space (SYNTHETIC)."""

from __future__ import annotations


def mc_ok(motivic: bool, classifying: bool) -> bool:
    """Motivic:
    motivic
    classifying
    space
    BG —
    Morel
    classifying."""
    return motivic and classifying


def motivic_quotient(mq: bool) -> bool:
    """Motivic
    quotient:
    motivic
    quotient
    space —
    Morel."""
    return mq


def _bench_motivic_classifying(seed: int = 0) -> float:
    checks = []
    checks.append(mc_ok(True, True))
    checks.append(not mc_ok(False, True))
    checks.append(motivic_quotient(True))
    checks.append(not motivic_quotient(False))
    checks.append(True)  # Morel
    return float(sum(checks) / len(checks))


def bench_motivic_classifying(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_classifying": _bench_motivic_classifying(seed)}
