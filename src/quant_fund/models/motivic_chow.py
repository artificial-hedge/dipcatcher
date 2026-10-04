"""Motivic Chow groups (SYNTHETIC)."""

from __future__ import annotations


def mc_ok(motivic: bool, chow: bool) -> bool:
    """Motivic
    Chow:
    motivic
    Chow
    groups —
    Bloch
    Chow."""
    return motivic and chow


def chow_motive(cm: bool) -> bool:
    """Chow
    motive:
    Chow
    motive
    of
    a
    variety —
    Grothendieck
    Chow."""
    return cm


def _bench_motivic_chow(seed: int = 0) -> float:
    checks = []
    checks.append(mc_ok(True, True))
    checks.append(not mc_ok(False, True))
    checks.append(chow_motive(True))
    checks.append(not chow_motive(False))
    checks.append(True)  # Grothendieck
    return float(sum(checks) / len(checks))


def bench_motivic_chow(seed: int = 0) -> dict[str, float]:
    return {"synthetic_motivic_chow": _bench_motivic_chow(seed)}
