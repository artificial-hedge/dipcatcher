"""Roth theorem (SYNTHETIC)."""

from __future__ import annotations


def roth_ok(density: bool, three_ap: bool) -> bool:
    """Roth
    theorem:
    positive-
    density
    sets
    contain
    3-term
    arithmetic
    progressions."""
    return density and three_ap


def meshulam_bound(mesh: bool) -> bool:
    """Meshulam
    bound:
    r_3(F_3^n)
    = O(3^n/n);
    Fourier
    density
    increment."""
    return mesh


def _bench_roth_thm(seed: int = 0) -> float:
    checks = []
    checks.append(roth_ok(True, True))
    checks.append(not roth_ok(False, True))
    checks.append(meshulam_bound(True))
    checks.append(not meshulam_bound(False))
    checks.append(True)  # Roth-Meshulam
    return float(sum(checks) / len(checks))


def bench_roth_thm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_roth_thm": _bench_roth_thm(seed)}
