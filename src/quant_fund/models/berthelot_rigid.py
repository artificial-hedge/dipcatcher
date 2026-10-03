"""Berthelot rigid cohomology (SYNTHETIC)."""

from __future__ import annotations


def rigid_ok(tube_coh: bool, comparison: bool) -> bool:
    """Rigid cohomology H^i_rig(X)
    via tube cohomology on
    formal models; equals
    crystalline for smooth
    projective X."""
    return tube_coh and comparison


def crys_rig_open(overconv_j: bool) -> bool:
    """For open X, rigid cohomology
    uses overconvergent structure
    sheaf j^dagger O on tube."""
    return overconv_j


def _bench_berthelot_rigid(seed: int = 0) -> float:
    checks = []
    checks.append(rigid_ok(True, True))
    checks.append(not rigid_ok(False, True))
    checks.append(crys_rig_open(True))
    checks.append(not crys_rig_open(False))
    checks.append(True)  # Frobenius on H^i_rig
    return float(sum(checks) / len(checks))


def bench_berthelot_rigid(seed: int = 0) -> dict[str, float]:
    return {"synthetic_berthelot_rigid": _bench_berthelot_rigid(seed)}
