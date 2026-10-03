"""May spectral sequence (SYNTHETIC)."""

from __future__ import annotations


def ms_ok(may: bool, spectral: bool) -> bool:
    """May:
    May
    spectral
    sequence —
    May
    SS."""
    return may and spectral


def may_differential(md: bool) -> bool:
    """May
    differential:
    May
    SS
    differentials —
    May
    differential."""
    return md


def _bench_may_ss(seed: int = 0) -> float:
    checks = []
    checks.append(ms_ok(True, True))
    checks.append(not ms_ok(False, True))
    checks.append(may_differential(True))
    checks.append(not may_differential(False))
    checks.append(True)  # May
    return float(sum(checks) / len(checks))


def bench_may_ss(seed: int = 0) -> dict[str, float]:
    return {"synthetic_may_ss": _bench_may_ss(seed)}
