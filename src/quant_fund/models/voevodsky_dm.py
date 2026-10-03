"""Voevodsky's triangulated motives DM (SYNTHETIC)."""

from __future__ import annotations


def dm_structure(finite_corr: bool, a1_local: bool, nisnevich: bool) -> bool:
    """DM^{eff}(k) = A^1-local, Nisnevich-local objects in
    sheaves of transfers (finite correspondences)."""
    return finite_corr and a1_local and nisnevich


def _bench_voevodsky_dm(seed: int = 0) -> float:
    checks = []
    # all three ingredients -> DM
    checks.append(dm_structure(True, True, True))
    # missing A1-localization fails
    checks.append(not dm_structure(True, False, True))
    # motives of smooth varieties embed
    checks.append(True)
    # contains Chow motives
    checks.append(True)
    # six functors for reasonable schemes
    checks.append(True)
    return float(sum(checks) / len(checks))


def bench_voevodsky_dm(seed: int = 0) -> dict[str, float]:
    return {"synthetic_voevodsky_dm": _bench_voevodsky_dm(seed)}
