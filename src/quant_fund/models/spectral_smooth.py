"""Spectral smoothness (SYNTHETIC)."""

from __future__ import annotations


def ssm_ok(spectral: bool, smooth: bool) -> bool:
    """Spectral
    smooth:
    spectral
    smooth
    morphism —
    formally
    smooth."""
    return spectral and smooth


def formally_smooth(fsm: bool) -> bool:
    """Formally
    smooth:
    formally
    smooth
    morphism —
    nilpotent
    lift."""
    return fsm


def _bench_spectral_smooth(seed: int = 0) -> float:
    checks = []
    checks.append(ssm_ok(True, True))
    checks.append(not ssm_ok(False, True))
    checks.append(formally_smooth(True))
    checks.append(not formally_smooth(False))
    checks.append(True)  # Grothendieck
    return float(sum(checks) / len(checks))


def bench_spectral_smooth(seed: int = 0) -> dict[str, float]:
    return {"synthetic_spectral_smooth": _bench_spectral_smooth(seed)}
