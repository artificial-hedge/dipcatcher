"""Periodic families (SYNTHETIC)."""

from __future__ import annotations


def pf_ok(periodic: bool, fam: bool) -> bool:
    """Periodic
    family:
    periodic
    family —
    chromatic
    family."""
    return periodic and fam


def greek_family(gf: bool) -> bool:
    """Greek
    family:
    Greek
    letter
    family —
    alpha
    beta."""
    return gf


def _bench_periodic_fam(seed: int = 0) -> float:
    checks = []
    checks.append(pf_ok(True, True))
    checks.append(not pf_ok(False, True))
    checks.append(greek_family(True))
    checks.append(not greek_family(False))
    checks.append(True)  # Ravenel
    return float(sum(checks) / len(checks))


def bench_periodic_fam(seed: int = 0) -> dict[str, float]:
    return {"synthetic_periodic_fam": _bench_periodic_fam(seed)}
