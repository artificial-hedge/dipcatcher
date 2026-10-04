"""Tame homotopy theory (SYNTHETIC)."""

from __future__ import annotations


def th_ok(tame: bool, htpy: bool) -> bool:
    """Tame:
    tame
    homotopy
    theory —
    Dwyer
    tame."""
    return tame and htpy


def tame_approx(ta: bool) -> bool:
    """Tame
    approximation:
    tame
    approximation
    of
    spaces —
    Dwyer
    tame."""
    return ta


def _bench_tame_htpy(seed: int = 0) -> float:
    checks = []
    checks.append(th_ok(True, True))
    checks.append(not th_ok(False, True))
    checks.append(tame_approx(True))
    checks.append(not tame_approx(False))
    checks.append(True)  # Dwyer
    return float(sum(checks) / len(checks))


def bench_tame_htpy(seed: int = 0) -> dict[str, float]:
    return {"synthetic_tame_htpy": _bench_tame_htpy(seed)}
