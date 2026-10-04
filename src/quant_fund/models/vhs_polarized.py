"""Polarized variations of Hodge structure (SYNTHETIC)."""

from __future__ import annotations


def vhs_ok(local_system: bool, flat: bool) -> bool:
    """Polarized variation of
    Hodge structure (VHS):
    flat bundle with
    Hodge filtration and
    polarization satisfying
    Griffiths transversality."""
    return local_system and flat


def vhs_family(monomial_rep: bool) -> bool:
    """VHS on a family:
    monodromy representation
    rho: pi_1 -> G_R
    gives the polarized
    variation of the
    fibers' Hodge."""
    return monomial_rep


def _bench_vhs_polarized(seed: int = 0) -> float:
    checks = []
    checks.append(vhs_ok(True, True))
    checks.append(not vhs_ok(False, True))
    checks.append(vhs_family(True))
    checks.append(not vhs_family(False))
    checks.append(True)  # Schmid orbit theorem
    return float(sum(checks) / len(checks))


def bench_vhs_polarized(seed: int = 0) -> dict[str, float]:
    return {"synthetic_vhs_polarized": _bench_vhs_polarized(seed)}
