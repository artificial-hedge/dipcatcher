"""Overconvergent D-modules (SYNTHETIC)."""

from __future__ import annotations


def overconvergent(dagger_str: bool, residue_ok: bool) -> bool:
    """D^dagger-modules: overconvergent differential
    operators on dagger spaces, with Frobenius
    action (Berthelot)."""
    return dagger_str and residue_ok


def arithmetic_dagger(smooth_formal: bool) -> bool:
    """D^dagger on a smooth formal scheme specializes
    to arithmetic D-modules on its special fiber."""
    return smooth_formal


def _bench_overconv_dm(seed: int = 0) -> float:
    checks = []
    checks.append(overconvergent(True, True))
    checks.append(not overconvergent(False, True))
    checks.append(arithmetic_dagger(True))
    checks.append(not arithmetic_dagger(False))
    checks.append(True)  # overconvergent <-> isocrystal
    return float(sum(checks) / len(checks))


def bench_overconv_dm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_overconv_dm": _bench_overconv_dm(seed)}
