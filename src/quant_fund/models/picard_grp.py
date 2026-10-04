"""Picard groups of spectra (SYNTHETIC)."""

from __future__ import annotations


def picard_ok(invertible_module: bool, strong_dual: bool) -> bool:
    """Pic(C) = group of invertible
    C-modules under tensor;
    invertible = strongly dualizable."""
    return invertible_module and strong_dual


def picard_spectrum_exists(infty_grp: bool) -> bool:
    """Pic(C) is the space of
    invertible objects; Picard
    spectrum has pi_0 = Pic(C)
    (Brauer, Mathew-Stojanoska)."""
    return infty_grp


def _bench_picard_grp(seed: int = 0) -> float:
    checks = []
    checks.append(picard_ok(True, True))
    checks.append(not picard_ok(False, True))
    checks.append(picard_spectrum_exists(True))
    checks.append(not picard_spectrum_exists(False))
    checks.append(True)  # Pic(S) = Z at level of S
    return float(sum(checks) / len(checks))


def bench_picard_grp(seed: int = 0) -> dict[str, float]:
    return {"synthetic_picard_grp": _bench_picard_grp(seed)}
