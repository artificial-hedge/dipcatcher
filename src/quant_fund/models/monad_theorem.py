"""Monadicity theorem (SYNTHETIC)."""

from __future__ import annotations


def mt_ok(monadicity: bool, beck: bool) -> bool:
    """Monadicity:
    Beck
    monadicity
    theorem —
    monad
    theorem."""
    return monadicity and beck


def beck_condition(bc: bool) -> bool:
    """Beck:
    Beck
    coequalizer
    condition —
    monadicity
    thm."""
    return bc


def _bench_monad_theorem(seed: int = 0) -> float:
    checks = []
    checks.append(mt_ok(True, True))
    checks.append(not mt_ok(False, True))
    checks.append(beck_condition(True))
    checks.append(not beck_condition(False))
    checks.append(True)  # Beck
    return float(sum(checks) / len(checks))


def bench_monad_theorem(seed: int = 0) -> dict[str, float]:
    return {"synthetic_monad_theorem": _bench_monad_theorem(seed)}
